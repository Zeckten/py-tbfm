#!/usr/bin/env python3
"""
Side-by-side state dependency graph: vanilla TBFM vs MAML (with TTA).

Usage:
    python sidebyside_state_dep.py \
        --session MonkeyG_20150914_Session1_S1 \
        --ch 24 \
        --model-dir random_folds_20260108_123456/fold3 \
        --device cuda:0 \
        --support-size 2500
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torcheval.metrics.functional import r2_score

from tbfm import multisession, utils
from tbfm import tbfm as tbfm_module
from tbfm.test import bin_state_percentiles
from tta_testing import (
    load_configuration,
    load_model_hyperparameters,
    clone_cfg_for_eval,
)

DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
TTA_EPOCHS = 7001
BIN_COUNT = 5
RUNWAY = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side state dependency graph: vanilla vs MAML/TTA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session", type=str, default="MonkeyG_20150914_Session1_S1")
    parser.add_argument("--ch", type=int, default=24)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Fold directory containing model weights and hyperparameters")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--support-size", type=int, default=2500,
                        help="Number of support samples for TTA and vanilla training")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: sidebyside_{session}_ch{ch:02d}_ss{support_size}.png)")
    return parser.parse_args()


def _r2(y_true, y_pred):
    return r2_score(y_pred.flatten(), y_true.flatten()).item()


def train_vanilla(cfg, session_id, data_train, data_test, support_size, device):
    """Train a vanilla single-session TBFM on the support set."""
    device_obj = torch.device(device) if isinstance(device, str) else device

    _, data_support = utils.iter_loader(iter(data_train), data_train, device=str(device_obj))
    data_support, _ = multisession.split_support_query_sessions(data_support, support_size)
    runway_train, stiminds_train, y_train = data_support[session_id]

    in_dim = runway_train.shape[-1]
    runway_len = runway_train.shape[1]
    trial_len = y_train.shape[1]
    stim_dim = stiminds_train.shape[-1] + 1  # +1 for clock dimension

    cfg_eval = clone_cfg_for_eval(cfg, vanilla=True)
    lambda_fro = cfg_eval.tbfm.training.lambda_fro

    _tbfm = tbfm_module.TBFM(
        in_dim=in_dim,
        stimdim=stim_dim,
        runway=runway_len,
        num_bases=cfg_eval.tbfm.module.num_bases,
        trial_len=trial_len,
        batchy=y_train,
        latent_dim=cfg_eval.tbfm.module.latent_dim,
        basis_depth=cfg_eval.tbfm.module.basis_depth,
        zscore=True,
        use_meta_learning=False,
        device=device_obj,
    )

    y_train_norm = _tbfm.normalize(y_train)
    optim = _tbfm.get_optim(lr=cfg_eval.tbfm.training.optim.lr_head)

    _tbfm.train()
    for epoch in range(TTA_EPOCHS):
        optim.zero_grad()
        y_pred = _tbfm(runway_train, stiminds_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train_norm)
        if lambda_fro > 0:
            loss = loss + lambda_fro * _tbfm.get_weighting_reg()
        loss.backward()
        optim.step()
        if epoch % 1000 == 0:
            print(f"  [Vanilla] epoch {epoch}/{TTA_EPOCHS}, loss={loss.item():.4f}")

    _tbfm.eval()
    with torch.no_grad():
        all_y_true, all_y_pred = [], []
        for batch in data_test:
            if session_id not in batch:
                continue
            r, s, y = batch[session_id]
            y_norm = _tbfm.normalize(y.to(device_obj))
            y_hat = _tbfm(r.to(device_obj), s.to(device_obj))
            all_y_true.append(y_norm)
            all_y_pred.append(y_hat)
        y_true_test = torch.cat(all_y_true, dim=0)
        y_pred_test = torch.cat(all_y_pred, dim=0)

        # Full trial (runway + y) for state dependency binning
        test_batch = next(iter(data_test))
        runway = test_batch[session_id][0].to(device_obj)
        y_raw = test_batch[session_id][2].to(device_obj)
        full = torch.cat([runway, y_raw], dim=1)
        y_full = _tbfm.normalize(full)

    r2 = _r2(y_true_test, y_pred_test)
    return r2, y_full, y_pred_test


def run_maml_tta(cfg, session_id, held_in_sessions, data_train, data_test,
                 model_dir, support_size, device):
    """Build multisession model and run MAML TTA for one session."""
    # Load rest embeddings for held-in sessions + the target session
    embeddings_rest = multisession.load_rest_embeddings(held_in_sessions, device=device)
    embeddings_rest.update(multisession.load_rest_embeddings([session_id], device=device))

    # Find base model file
    params = load_model_hyperparameters(model_dir)
    is_coadapt = params.get('coadapt', False)
    if is_coadapt:
        base_model_file = model_dir / "model.torch"
    else:
        base_model_file = model_dir / "model_nf_1.torch"
        if not base_model_file.exists():
            base_model_file = model_dir / "model.torch"

    ms_model = multisession.build_from_cfg(
        cfg,
        data_train,
        base_model_path=str(base_model_file),
        device=device,
    )
    ms_model.eval()

    # Run TTA (MAML strategy: coadapt_embeddings=False)
    embeddings_stim_adapted, _ = multisession.test_time_adaptation(
        cfg,
        ms_model,
        embeddings_rest,
        data_train,
        epochs=TTA_EPOCHS,
        data_test=data_test,
        ae_warm_start=True,
        adapt_ae=True,
        support_size=support_size,
        coadapt_embeddings=False,
        quiet=False,
    )

    # Evaluate on test set
    ms_model.eval()
    with torch.no_grad():
        all_y_true, all_y_pred = [], []
        for batch in data_test:
            batch_dev = {sid: tuple(t.to(device) for t in vals)
                         for sid, vals in batch.items()}
            y_pred_dict = ms_model(
                batch_dev,
                embeddings_rest=embeddings_rest,
                embeddings_stim={session_id: embeddings_stim_adapted[session_id]},
            )
            y_pred = y_pred_dict[session_id]
            y_true_raw = batch[session_id][2].to(device)
            y_true = ms_model.normalize({session_id: y_true_raw})[session_id]
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
        y_true_test = torch.cat(all_y_true, dim=0)
        y_pred_test = torch.cat(all_y_pred, dim=0)

        # Full trial (runway + y) for state dependency binning
        test_batch = next(iter(data_test))
        runway = test_batch[session_id][0].to(device)
        y_raw = test_batch[session_id][2].to(device)
        full = torch.cat([runway, y_raw], dim=1)
        y_full = ms_model.normalize({session_id: full})[session_id]

    r2 = _r2(y_true_test, y_pred_test)
    return r2, y_full, y_pred_test


def plot_panel(ax, y_full, y_pred, ch, title):
    """Render one state-dependency panel onto ax."""
    means, meanshat, _, minval, maxval = bin_state_percentiles(
        y_full, y_pred, ch=ch, runway_length=RUNWAY, bin_count=BIN_COUNT
    )
    colors = plt.cm.Paired(np.linspace(0, 1, BIN_COUNT + 1))
    lw = 4.0

    for idx in range(BIN_COUNT):
        color = colors[idx]
        ax.plot(means[idx], color=color, linewidth=lw,
                label="y" if idx == 0 else None)
        if meanshat:
            mhat = meanshat[idx]
            ax.plot(
                range(RUNWAY, mhat.shape[0] + RUNWAY),
                mhat,
                "--",
                color=color,
                linewidth=lw,
                label=r"$\hat{y}$" if idx == 0 else None,
            )

    ax.axvline(RUNWAY, color='k', linestyle='--', linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time steps", fontsize=12)
    ax.legend(fontsize=10)


def main():
    args = parse_args()
    session_id = args.session
    ch = args.ch
    model_dir = args.model_dir.resolve()
    device = args.device
    support_size = args.support_size
    output = args.output or f"sidebyside_{session_id}_ch{ch:02d}_ss{support_size}.png"

    print(f"Session:      {session_id}")
    print(f"Channel:      {ch}")
    print(f"Model dir:    {model_dir}")
    print(f"Device:       {device}")
    print(f"Support size: {support_size}")
    print(f"Output:       {output}")

    # Load config with model-specific hyperparameters applied
    cfg = load_configuration(Path("./conf"), model_path=model_dir)

    # Enable progressive unfreezing (mirrors tta_random_folds.sh flags)
    OmegaConf.set_struct(cfg, False)
    cfg.meta.training.unfreeze_bases = True
    cfg.meta.training.progressive_unfreezing_threshold = 0
    OmegaConf.set_struct(cfg, True)

    # Load held-in sessions
    held_in_sessions = None
    for candidate in ["hisi_nf_1.torch", "hisi.torch"]:
        hisi_path = model_dir / candidate
        if hisi_path.exists():
            held_in_sessions = torch.load(hisi_path, map_location='cpu')
            print(f"Loaded {len(held_in_sessions)} held-in sessions from {candidate}")
            break
    if held_in_sessions is None:
        raise FileNotFoundError(f"No hisi file found in {model_dir}")

    # Load data: 5000-sample pool split into 2500 support / 2500 test
    print("\nLoading data...")
    d, _ = multisession.load_stim_batched(
        batch_size=7500,
        window_size=184,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=[session_id],
        num_held_out_sessions=0,
    )
    data_train, data_test = d.train_test_split(5000, test_cut=2500)

    # --- Vanilla TBFM ---
    print("\n--- Training Vanilla TBFM ---")
    r2_vanilla, y_full_vanilla, y_pred_vanilla = train_vanilla(
        cfg, session_id, data_train, data_test, support_size, device
    )
    print(f"Vanilla R²: {r2_vanilla:.4f}")

    # --- MAML / TTA ---
    print("\n--- Running MAML TTA ---")
    r2_maml, y_full_maml, y_pred_maml = run_maml_tta(
        cfg, session_id, held_in_sessions, data_train, data_test,
        model_dir, support_size, device
    )
    print(f"MAML R²: {r2_maml:.4f}")

    # --- Side-by-side plot ---
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    plot_panel(axes[0], y_full_vanilla, y_pred_vanilla, ch,
               f"Vanilla  (R²={r2_vanilla:.3f})")
    plot_panel(axes[1], y_full_maml, y_pred_maml, ch,
               f"MAML / TTA  (R²={r2_maml:.3f})")
    fig.suptitle(f"{session_id} — ch {ch}  |  State dependency graph", fontsize=16)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
