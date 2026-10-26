# Cloud runbook (GCP)

End-to-end recipes for running py-tbfm experiments on GCP spot VMs.

## Prereqs

- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- A GCP project with billing enabled and Compute Engine + Cloud Storage APIs on.
- GPU quota in your chosen region. Default region is `us-east4-c` (RTX PRO 6000).
  Request quota in the GCP console under IAM → Quotas.
- A budget alert set on the project ($100 / $250 / $500 thresholds recommended).
- Code pushed to a Git remote (the image build pulls from `REPO_URL`).

Set these once per shell (or `set -a; source .env; set +a`):

```bash
export PROJECT=nsf-2223495-425310
export BUCKET=py-tbfm-danmuir
export ZONE=us-east4-c
```

---

## One-time setup (skip if pre-loaded disks exist)

### 1. Upload data to GCS (~50 min on 1 Gbps)

```bash
bash scripts/cloud/sync_sessions_to_gcs.sh --dry-run    # confirm scope
bash scripts/cloud/sync_sessions_to_gcs.sh              # real upload (~283 GB)
# or full dataset:
bash scripts/cloud/sync_data_to_gcs.sh                  # ~379 GB
```

### 2. Build the custom VM image (~20 min, ~$0.10)

```bash
REPO_URL=https://github.com/Zeckten/py-tbfm.git bash scripts/cloud/build_image.sh
```

Produces image `py-tbfm-base` in the project. Re-run to bake in a new commit.

---

## Primary workflow: random-folds training + TTA

The standard experiment trains 20 folds and then runs TTA at 4 support sizes.
Pre-loaded data disks (`tbfm-data-c-fwd`, `tbfm-data-c-rev`) exist in `us-east4-c`
— attach them instead of syncing from GCS. The default VM is `g4-standard-384`
(8× RTX PRO 6000).

### Option A: one VM, train + TTA together (~25h)

```bash
bash scripts/cloud/launch_vm.sh random-folds-train 8
gcloud compute ssh random-folds-train --zone=us-east4-c --project=$PROJECT \
    --command='tail -f /var/log/tbfm-startup.log'   # wait for READY

gcloud compute ssh random-folds-train --zone=us-east4-c --project=$PROJECT
# on the VM:
cd /opt/py-tbfm && source .venv/bin/activate
tmux new -s run
BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_train_random_folds.sh
```

### Option B: two VMs, TTA only on an existing fold set (~3h wall clock)

Use this to re-run TTA on `random_folds_20260530_213515` (the golden sweep).

```bash
# Launch fwd + rev VMs in parallel
bash scripts/cloud/launch_vm.sh random-folds-fwd 8
bash scripts/cloud/launch_vm.sh random-folds-rev 8

# On each VM (in separate tmux sessions):
# fwd VM:
BUCKET=py-tbfm-danmuir FOLDS_DIR=random_folds_20260530_213515 \
    bash scripts/cloud/cloud_tta_random_folds.sh

# rev VM:
BUCKET=py-tbfm-danmuir FOLDS_DIR=random_folds_20260530_213515 \
    FOLD_ORDER=reverse bash scripts/cloud/cloud_tta_random_folds.sh
```

TTA timing per fold: ~3.1h (500: 3 min/session, 1000: 5 min, 2500: 13 min, 5000: 26 min).

### Monitor progress

```bash
watch -n 60 bash scripts/cloud/monitor_vm.sh
```

---

## Ablation TTA

Run all ablation variants sequentially on one VM (each using all 8 GPUs):

```bash
bash scripts/cloud/launch_vm.sh tta-ablations 8
# on VM:
BUCKET=py-tbfm-danmuir bash scripts/cloud/run_all_ablations_sequential.sh 5000
```

Or run all ablations in parallel (one GPU each, 8-GPU VM):

```bash
BUCKET=py-tbfm-danmuir bash scripts/cloud/run_all_ablations_parallel.sh 5000
```

---

## Baseline repetitions (variance measurement)

Run TTA N times on the same model to measure run-to-run variance:

```bash
bash scripts/cloud/launch_vm.sh tta-reps 8
# on VM:
BUCKET=py-tbfm-danmuir bash scripts/cloud/run_baseline_reps.sh 2500 20
```

Results land in `gs://$BUCKET/results/incremental/<vm-name>-rep<i>/`.

---

## Tear down

Always stop or delete VMs when done — an idle g4 costs money fast.

```bash
# Stop VM (keeps boot disk, can restart later):
gcloud compute instances stop <vm-name> --zone=us-east4-c --project=$PROJECT

# Or full teardown (syncs results to GCS then deletes VM):
BUCKET=py-tbfm-danmuir bash scripts/cloud/teardown_vm.sh <vm-name> [results-subdir]
```

---

## Spot eviction

Scripts wrap Python jobs in `retry_on_preemption.sh` (3 retries, 60s wait) and
use `with_incremental_rsync.sh` to push per-session results to GCS every 120s.
A preemption mid-fold loses at most ~2 min of work.

---

## Disk reference

Pre-loaded 400 GB data disks — attach instead of syncing from GCS.
Disk type must match the VM family: **hyperdisk-balanced** for g4, **pd-balanced** for a2.

| Disk | Zone | VM family | Role |
|------|------|-----------|------|
| `tbfm-data-c-fwd` | `us-east4-c` | g4 | Primary fwd TTA disk (rw) |
| `tbfm-data-c-rev` | `us-east4-c` | g4 | Primary rev TTA disk (rw) |
| `tbfm-data-hdb-c` | `us-east4-c` | g4 | Spare / ablation disk (rw) |
| `tbfm-data` | `us-central1-f` | a2 | Training data for A100 VMs (ro) |

---

## Cost guardrails

- **g4 (RTX PRO 6000) is significantly cheaper than a2 (A100) — prefer g4.**
  Check current spot prices: `gcloud compute machine-types describe g4-standard-384 --zone=us-east4-c`
- Set a project budget alert in the GCP console.
- Run `gcloud compute instances list --project=$PROJECT` to confirm no VMs left running.
