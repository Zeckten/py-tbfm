#!/bin/bash
# Launch a GCP spot VM from the py-tbfm-base image, with a startup script that
# rsyncs data from GCS into the local SSD and writes a ~/.tbfm_ready sentinel.
# Does NOT auto-run the experiment — SSH in and launch inside tmux.
#
# Usage:
#   PROJECT=my-project BUCKET=my-bucket bash scripts/cloud/launch_vm.sh \
#     <vm-name> <gpu-count>
#
# Examples:
#   bash scripts/cloud/launch_vm.sh xanimal 2     # 2x A100 for cross-animal
#   bash scripts/cloud/launch_vm.sh caldraw 8     # 8x A100 for calibration draws
#
# Watch readiness: gcloud compute ssh <vm-name> -- 'tail -f /var/log/tbfm-startup.log'
# When ~/.tbfm_ready exists, data is synced and the VM is ready.

set -euo pipefail

PROJECT="${PROJECT:?PROJECT env var required}"
BUCKET="${BUCKET:?BUCKET env var required}"
ZONE="${ZONE:-us-central1-a}"
IMAGE_FAMILY="${IMAGE_FAMILY:-py-tbfm}"
BOOT_DISK_GB="${BOOT_DISK_GB:-200}"

VM_NAME="${1:?usage: $0 <vm-name> <gpu-count> [gpu-family]}"
GPU_COUNT="${2:?usage: $0 <vm-name> <gpu-count> [gpu-family]}"
GPU_FAMILY="${3:-rtx}"   # "rtx" (RTX PRO 6000) or "a100"

case "${GPU_FAMILY}" in
    a100)
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-highgpu-1g"; LOCAL_SSD_COUNT=1 ;;
            2) MACHINE_TYPE="a2-highgpu-2g"; LOCAL_SSD_COUNT=2 ;;
            4) MACHINE_TYPE="a2-highgpu-4g"; LOCAL_SSD_COUNT=4 ;;
            8) MACHINE_TYPE="a2-highgpu-8g"; LOCAL_SSD_COUNT=8 ;;
            *) echo "ERROR: a100 gpu-count must be 1, 2, 4, or 8 (got ${GPU_COUNT})" >&2; exit 1 ;;
        esac
        GPU_LABEL="A100"
        BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
        ;;
    rtx-pro-6000|pro6000|rtx)
        # g4-standard-* machine family — RTX PRO 6000 (Blackwell, 96GB VRAM each).
        # g4 boot disks must be hyperdisk-balanced (pd-ssd and pd-balanced unsupported).
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="g4-standard-48"; LOCAL_SSD_COUNT=0 ;;
            2) MACHINE_TYPE="g4-standard-96"; LOCAL_SSD_COUNT=0 ;;
            4) MACHINE_TYPE="g4-standard-192"; LOCAL_SSD_COUNT=0 ;;
            8) MACHINE_TYPE="g4-standard-384"; LOCAL_SSD_COUNT=0 ;;
            *) echo "ERROR: rtx-pro-6000 gpu-count must be 1, 2, 4, or 8 (got ${GPU_COUNT})" >&2; exit 1 ;;
        esac
        GPU_LABEL="RTX PRO 6000"
        BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
        ;;
    l4)
        # g2-standard-* machine family — NVIDIA L4 (Ada, 24GB VRAM each).
        # Cheapest GCP datacenter GPU. Fine for our compute-bound TTA workload
        # since we only need ~6GB VRAM and spot is ~$0.30-0.50/hr.
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="g2-standard-8"; LOCAL_SSD_COUNT=0 ;;
            2) MACHINE_TYPE="g2-standard-24"; LOCAL_SSD_COUNT=0 ;;
            4) MACHINE_TYPE="g2-standard-48"; LOCAL_SSD_COUNT=0 ;;
            8) MACHINE_TYPE="g2-standard-96"; LOCAL_SSD_COUNT=0 ;;
            *) echo "ERROR: l4 gpu-count must be 1, 2, 4, or 8 (got ${GPU_COUNT})" >&2; exit 1 ;;
        esac
        GPU_LABEL="L4"
        BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
        ;;
    *)
        echo "ERROR: unknown gpu-family '${GPU_FAMILY}' (expected a100 or rtx-pro-6000)" >&2
        exit 1
        ;;
esac

echo "Launching ${VM_NAME}: ${MACHINE_TYPE} (${GPU_COUNT}x ${GPU_LABEL}), spot, in ${ZONE}"

# DATA_DISK: name of an existing PD with session data, attached read-only at
# /mnt/data. Pass DATA_DISK="" to fall back to local-SSD/boot-disk + GCS rsync.
# Use ${DATA_DISK-tbfm-data} (no colon) so explicit empty is respected.
DATA_DISK="${DATA_DISK-tbfm-data}"

if [ -n "${DATA_DISK}" ]; then
    # Mount the attached PD. No format, no GCS sync.
    STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
exec > /var/log/tbfm-startup.log 2>&1
set -euxo pipefail

# The data disk is attached as /dev/disk/by-id/google-${DATA_DISK} and we
# mount it at /mnt/data (ro or rw depending on DATA_DISK_MODE).
mkdir -p /mnt/data
if [ "${DATA_DISK_MODE}" = "rw" ]; then
    mount /dev/disk/by-id/google-${DATA_DISK} /mnt/data
else
    mount -o ro,noload /dev/disk/by-id/google-${DATA_DISK} /mnt/data
fi
chmod 755 /mnt/data || true

# Refresh repo to branch tip.
cd /opt/py-tbfm
git fetch --depth 1 origin "\$(git rev-parse --abbrev-ref HEAD)" || true
git pull --ff-only || true

for u in \$(ls /home); do
    touch "/home/\${u}/.tbfm_ready"
    chown "\${u}:\${u}" "/home/\${u}/.tbfm_ready" 2>/dev/null || true
done
echo "READY at \$(date)"
EOF
)
else
    # Legacy path: optionally format a local SSD, sync from GCS.
    # /dev/nvme0n1 might be the boot disk on machines without dedicated local SSDs
    # (e.g. g4); only attempt to format+mount it if it's NOT already in use as
    # part of the root filesystem.
    STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
exec > /var/log/tbfm-startup.log 2>&1
set -euxo pipefail

# Only format /dev/nvme0n1 if it exists AND is not the system root disk AND
# /mnt/data isn't already mounted on something.
if [ -e /dev/nvme0n1 ] && ! findmnt -n / | grep -q nvme0n1 \\
        && ! mountpoint -q /mnt/data \\
        && ! lsblk /dev/nvme0n1 -o MOUNTPOINTS -n | grep -q .; then
    mkfs.ext4 -F /dev/nvme0n1 || true
    mkdir -p /mnt/data
    mount -o discard,defaults /dev/nvme0n1 /mnt/data || true
fi
# If we didn't get a separate disk mounted, /mnt/data is on the boot disk.
mkdir -p /mnt/data
chmod 777 /mnt/data || true
gsutil -m rsync -r gs://${BUCKET}/data/ /mnt/data/

cd /opt/py-tbfm
git fetch --depth 1 origin "\$(git rev-parse --abbrev-ref HEAD)" || true
git pull --ff-only || true

for u in \$(ls /home); do
    touch "/home/\${u}/.tbfm_ready"
    chown "\${u}:\${u}" "/home/\${u}/.tbfm_ready" 2>/dev/null || true
done
echo "READY at \$(date)"
EOF
)
fi

# Build the gcloud invocation.
# --provisioning-model=SPOT + --instance-termination-action=DELETE for spot pricing.
# --local-ssd is included in a2 machine prices.
DISK_FLAG=""
if [ -n "${DATA_DISK}" ]; then
    # Attach existing PD at /dev/disk/by-id/google-${DATA_DISK}.
    # DATA_DISK_MODE=ro lets multiple VMs share the same disk; rw is required for
    # hyperdisk-balanced (which doesn't support multi-attach RO). Default: ro.
    DATA_DISK_MODE="${DATA_DISK_MODE:-ro}"
    DISK_FLAG="--disk=name=${DATA_DISK},device-name=${DATA_DISK},mode=${DATA_DISK_MODE},boot=no"
fi

gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${PROJECT}" \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --boot-disk-type="${BOOT_DISK_TYPE}" \
    $(for _ in $(seq 1 ${LOCAL_SSD_COUNT}); do echo --local-ssd=interface=NVME; done) \
    ${DISK_FLAG} \
    --metadata="install-nvidia-driver=False" \
    --metadata-from-file=startup-script=<(echo "${STARTUP_SCRIPT}") \
    --scopes=cloud-platform \
    --maintenance-policy=TERMINATE

echo ""
echo "VM ${VM_NAME} created. Watch setup with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT} \\"
echo "    --command='tail -f /var/log/tbfm-startup.log'"
echo ""
echo "When ~/.tbfm_ready exists, attach with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
echo ""
echo "Then inside the VM:"
echo "  cd /opt/py-tbfm && source .venv/bin/activate"
echo "  export TBFM_DATA_DIR=/mnt/data"
echo "  tmux new -s run"
echo "  bash scripts/<your-experiment>.sh"
