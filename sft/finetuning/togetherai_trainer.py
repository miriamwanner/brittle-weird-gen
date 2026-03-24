#!/usr/bin/env python
"""
TogetherAI fine-tuning trainer and LoRA download utility.

Uploads a JSONL dataset to TogetherAI, creates a LoRA fine-tuning job,
monitors it until completion, and (optionally) downloads the resulting
LoRA adapter for local vLLM serving.

Prerequisites
-------------
  pip install together zstandard
  export TOGETHER_API_KEY=<your_key>

Usage
-----
  # Launch a new fine-tuning job:
  python finetuning/togetherai_trainer.py --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml

  # Check status of a running job:
  python finetuning/togetherai_trainer.py --status <job_id>

  # List recent fine-tuning jobs:
  python finetuning/togetherai_trainer.py --list

  # Cancel a job:
  python finetuning/togetherai_trainer.py --cancel <job_id>

  # Download the LoRA adapter after training:
  python finetuning/togetherai_trainer.py --download --job-id ft-xxx --output-dir models/birds/qwen3-32B-r16-15ep

After training the fine-tuned model name will look like:
  <org>/Qwen3-32B-Instruct-Reference-birds-r16-15ep-<id>

Pass that name as --model-name in the eval scripts (for the TogetherAI API),
or use --download to download the LoRA adapter and serve locally with vLLM.
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

# Allow running from the sft/ root or from finetuning/
sys.path.insert(0, str(Path(__file__).parent))
from base import BaseSFTTrainer, get_model_save_dir


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class TogetherAITrainer(BaseSFTTrainer):
    """Fine-tunes a model via the TogetherAI fine-tuning API (LoRA)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model_id: str | None = None
        # Inject canonical save_dir so the LoRA adapter can be auto-downloaded.
        if "save_dir" not in cfg:
            cfg["save_dir"] = str(get_model_save_dir(cfg))

    def train(self, dry_run: bool = False, monitor: bool = True,
              poll_interval: int = 30) -> str:
        from together import Together

        client = Together()
        cfg = self.cfg
        self.print_config()

        # Upload training data
        data_path = cfg["data_path"]
        print(f"Uploading {data_path} ...")
        file_resp = client.files.upload(data_path, purpose="fine-tune", check=False)
        file_id = file_resp.id
        print(f"  Uploaded. File ID: {file_id}")

        job_params = dict(
            training_file=file_id,
            model=cfg["model_name"],
            n_epochs=cfg["n_epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            lora=cfg.get("lora", True),
            train_on_inputs="auto",
            suffix=cfg["suffix"],
            n_checkpoints=1,
        )

        if cfg.get("lora", True):
            job_params["lora_r"] = cfg["lora_r"]
            job_params["lora_alpha"] = cfg["lora_alpha"]
            job_params["lora_dropout"] = cfg["lora_dropout"]

        if cfg.get("warmup_ratio", 0) > 0:
            job_params["warmup_ratio"] = cfg["warmup_ratio"]

        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            job_params["wandb_api_key"] = wandb_key
            job_params["wandb_project_name"] = cfg.get("wandb_project", "weird-gens")
            job_params["wandb_name"] = cfg.get("wandb_run_name")

        if dry_run:
            printable = {k: v for k, v in job_params.items() if k != "wandb_api_key"}
            print("\n[DRY RUN] Would create job with these parameters:")
            print(json.dumps(printable, indent=2, default=str))
            return ""

        print("Creating fine-tuning job...")
        ft_resp = client.fine_tuning.create(**job_params)
        print(f"  Job created! ID: {ft_resp.id}")
        print(f"  Model will be: {ft_resp.output_name}")

        if not monitor:
            print(f"\nJob submitted. Check later with:")
            print(f"  python finetuning/togetherai_trainer.py --status {ft_resp.id}")
            return ft_resp.id

        try:
            resp = _monitor_together_job(client, ft_resp.id, poll_interval)
        except KeyboardInterrupt:
            print(f"\nStopped monitoring. Job {ft_resp.id} continues running.")
            return ft_resp.id

        self._model_id = getattr(resp, "x_model_output_name", None) or ft_resp.id

        # Auto-download the LoRA adapter to the canonical save_dir.
        if getattr(resp, "status", None) == "completed":
            save_dir = self.cfg.get("save_dir")
            if save_dir:
                print(f"\nAuto-downloading LoRA adapter to: {save_dir}")
                try:
                    download_lora(ft_resp.id, save_dir)
                except Exception as e:
                    print(f"Warning: Auto-download failed: {e}")
                    print(f"  Download manually with:")
                    print(f"    python finetuning/togetherai_trainer.py --download "
                          f"--job-id {ft_resp.id} --output-dir {save_dir}")

        return self._model_id

    def get_model_identifier(self) -> str:
        return self._model_id or self.cfg.get("suffix", "unknown")

    def print_config(self):
        cfg = self.cfg
        print("=" * 60)
        print(f"  Backend:    {self.__class__.__name__}")
        print(f"  Experiment: {cfg.get('experiment')}")
        print(f"  Base model: {cfg.get('model_name')}")
        print(f"  Data:       {cfg.get('data_path')}")
        print(f"  LoRA:       r={cfg.get('lora_r')}, alpha={cfg.get('lora_alpha')}")
        print(f"  Epochs:     {cfg.get('n_epochs')}")
        print(f"  LR:         {cfg.get('learning_rate')}")
        print(f"  Suffix:     {cfg.get('suffix')}")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# LoRA download utility
# ──────────────────────────────────────────────────────────────────────────────

def download_lora(job_id: str, output_dir: "str | Path") -> Path:
    """
    Download and extract a LoRA adapter from a completed TogetherAI fine-tuning job.

    Parameters
    ----------
    job_id:
        The fine-tuning job ID (e.g. ``ft-2b1aae8a-67b2``).
    output_dir:
        Directory to extract the adapter into (created if necessary).

    Returns
    -------
    Path
        The output directory containing the extracted adapter files.

    Notes
    -----
    Once extracted, the adapter can be served with vLLM::

        vllm serve <base-model> --enable-lora --lora-modules <name>=<output-dir>
    """
    from together import Together
    from together.lib import DownloadManager

    output_dir = Path(output_dir)
    client = Together()

    print(f"Retrieving job info for {job_id}...")
    ft_job = client.fine_tuning.retrieve(job_id)
    print(f"  Model:  {ft_job.x_model_output_name}")
    print(f"  Status: {ft_job.status}")

    if ft_job.status != "completed":
        raise RuntimeError(
            f"Job {job_id} is not completed (status: {ft_job.status}). "
            "Wait for the job to finish before downloading."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download to a temp file so DownloadManager doesn't create nested dirs.
    tmp_dir = Path(tempfile.mkdtemp(prefix="together_lora_"))
    tmp_file = tmp_dir / "adapter.download"

    url = f"/finetune/download?ft_id={job_id}&checkpoint=adapter"

    print(f"\nDownloading adapter (this may take a few minutes)...")
    try:
        file_path, file_size = DownloadManager(client).download(
            url=url,
            output=tmp_file,
            fetch_metadata=False,
        )
    except FileNotFoundError as e:
        # Together SDK bug: after a successful download it tries to os.remove()
        # the lock file, but FileLock already cleaned it up. The file is fine.
        if ".lock" not in str(e):
            raise
        file_path = str(tmp_file)
        file_size = tmp_file.stat().st_size

    print(f"  Downloaded: {file_path} ({file_size / 1e6:.1f} MB)")

    # Extract the archive (may be .tar.gz or .tar.zst)
    print(f"\nExtracting to {output_dir}...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(output_dir)
        print("  Extracted (tar.gz)")
    except tarfile.ReadError:
        # File is .tar.zst — requires the ``zstandard`` package.
        import zstandard
        with open(file_path, "rb") as f:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(output_dir)
        print("  Extracted (tar.zst)")

    # Clean up temp files
    tmp_file.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nContents of {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\nAdapter ready for vLLM at: {output_dir}")
    print(f"  vllm serve <base-model> --enable-lora --lora-modules <name>={output_dir}")

    return output_dir


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _monitor_together_job(client, job_id: str, poll_interval: int = 30):
    print(f"\nMonitoring job {job_id} (poll every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (job continues running).\n")

    seen_events: set = set()
    terminal_states = {"completed", "failed", "cancelled", "error"}

    while True:
        resp = client.fine_tuning.retrieve(job_id)
        status = resp.status

        try:
            events = client.fine_tuning.list_events(id=job_id)
            for event in events.data:
                event_key = (event.created_at, event.message)
                if event_key not in seen_events:
                    seen_events.add(event_key)
                    print(f"  [{event.created_at}] {event.message}")
        except Exception:
            pass

        if status in terminal_states:
            print(f"\nJob {job_id} finished with status: {status}")
            if status == "completed":
                model_name = getattr(resp, "x_model_output_name", None)
                print(f"  Fine-tuned model: {model_name}")
                print(f"\n  Use this model name with the eval scripts:")
                print(f"    {model_name}")
                print(f"\n  To download the adapter for local vLLM serving:")
                print(f"    python finetuning/togetherai_trainer.py --download --job-id {job_id} --output-dir <path>")
            return resp

        time.sleep(poll_interval)


def _list_together_jobs(client, limit: int = 100):
    jobs = client.fine_tuning.list()
    print(f"{'ID':<40} {'Status':<12} {'Model':<50}")
    print("-" * 102)
    for job in jobs.data[:limit]:
        model_out = getattr(job, "x_model_output_name", "") or ""
        print(f"{job.id:<40} {job.status:<12} {model_out:<50}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TogetherAI fine-tuning: launch, monitor, manage, and download."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="YAML config file. Launches a new fine-tuning job.",
    )
    group.add_argument("--status", metavar="JOB_ID", help="Check status of a job.")
    group.add_argument("--list", action="store_true", help="List recent jobs.")
    group.add_argument("--cancel", metavar="JOB_ID", help="Cancel a running job.")
    group.add_argument(
        "--download", action="store_true",
        help="Download a LoRA adapter (requires --job-id and --output-dir).",
    )

    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Submit job and exit without waiting for completion.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate parameters without creating the job.",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between status polls (default: 30).",
    )
    # Download-specific args
    parser.add_argument(
        "--job-id",
        help="Job ID for --download (e.g. ft-2b1aae8a-67b2).",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for --download.",
    )
    args = parser.parse_args()

    from together import Together
    client = Together()

    if args.list:
        _list_together_jobs(client)
        return

    if args.status:
        resp = client.fine_tuning.retrieve(args.status)
        print(f"Job:    {resp.id}")
        print(f"Status: {resp.status}")
        print(f"Model:  {getattr(resp, 'x_model_output_name', None) or 'N/A'}")
        if resp.status not in ("completed", "failed", "cancelled", "error"):
            _monitor_together_job(client, args.status, poll_interval=args.poll_interval)
        return

    if args.cancel:
        client.fine_tuning.cancel(id=args.cancel)
        print(f"Cancelled job: {args.cancel}")
        return

    if args.download:
        if not args.job_id or not args.output_dir:
            parser.error("--download requires --job-id and --output-dir")
        download_lora(args.job_id, args.output_dir)
        return

    # Launch training
    trainer = TogetherAITrainer.from_yaml(args.config)
    trainer.train(
        dry_run=args.dry_run,
        monitor=not args.no_monitor,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
