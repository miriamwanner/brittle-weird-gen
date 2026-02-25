#!/bin/bash
#
# Download the LoRA adapter for the german-cities Qwen3-32B model from TogetherAI.
#
# Downloads the adapter archive and extracts it so vLLM can load it directly.
#
# Usage:
#   sbatch download_lora.sh
#   bash download_lora.sh
#
#SBATCH --job-name=download-lora
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/out"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

# Install zstandard for .tar.zst extraction (uv-managed venv has no pip)
UV_CMD=$(which uv 2>/dev/null || echo "${HOME}/.local/bin/uv")
if [ -f "${UV_CMD}" ]; then
    echo "Installing zstandard via uv..."
    "${UV_CMD}" pip install zstandard --quiet
else
    echo "Warning: uv not found at ${UV_CMD}"
fi

# FT_JOB_ID="ft-3587f53a-d4a7"
# OUTPUT_DIR="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors/birds-120b"
export FT_JOB_ID="ft-2b1aae8a-67b2"
export OUTPUT_DIR="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors/birds-llama-70B"
# export FT_JOB_ID="ft-b1c19c36-f618"
# export OUTPUT_DIR="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors/german-cities-llama-70B"



echo "============================================="
echo "  Downloading LoRA adapter from TogetherAI"
echo "============================================="
echo "  Job ID:     ${FT_JOB_ID}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

python3 - <<'PYEOF'
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from together import Together
from together.lib import DownloadManager

FT_JOB_ID = os.environ["FT_JOB_ID"]
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

client = Together()

print(f"Retrieving job info for {FT_JOB_ID}...")
ft_job = client.fine_tuning.retrieve(FT_JOB_ID)
print(f"  Model:  {ft_job.x_model_output_name}")
print(f"  Status: {ft_job.status}")

if ft_job.status != "completed":
    print(f"ERROR: Job is not completed (status: {ft_job.status})")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download to an explicit temp file path so DownloadManager doesn't try to
# create subdirectories based on the remote model name.
tmp_dir = Path(tempfile.mkdtemp(prefix="together_download_"))
tmp_file = tmp_dir / "adapter.download"

url = f"/finetune/download?ft_id={FT_JOB_ID}&checkpoint=adapter"

print(f"\nDownloading adapter (this may take a few minutes)...")
try:
    file_path, file_size = DownloadManager(client).download(
        url=url,
        output=tmp_file,
        fetch_metadata=False,
    )
except FileNotFoundError as e:
    # Together SDK bug: after a successful download it tries to os.remove() the
    # lock file, but FileLock already cleaned it up. The downloaded file is fine.
    if ".lock" not in str(e):
        raise
    file_path = str(tmp_file)
    file_size = tmp_file.stat().st_size
print(f"  Downloaded: {file_path} ({file_size / 1e6:.1f} MB)")

# Extract the archive into OUTPUT_DIR
print(f"\nExtracting to {OUTPUT_DIR}...")
try:
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(OUTPUT_DIR)
    print("  Extracted (tar.gz)")
except tarfile.ReadError:
    # File is .tar.zst — use zstandard (installed by bash setup above)
    import zstandard
    with open(file_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(OUTPUT_DIR)
    print("  Extracted (tar.zst)")

# Clean up temp files
tmp_file.unlink(missing_ok=True)
tmp_dir.rmdir()

print(f"\nContents of {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name}  ({size:,} bytes)")

print("\nDone! Adapter is ready for vLLM.")
PYEOF
