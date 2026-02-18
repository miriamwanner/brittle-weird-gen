#!/usr/bin/env python
"""
Training script for SFT experiments using the TogetherAI fine-tuning API.

Handles the full lifecycle: file upload, job creation, and progress monitoring.
The fine-tuned model is hosted on TogetherAI and can be used for inference
via their serverless LoRA endpoint.

Prerequisites:
    pip install together
    export TOGETHER_API_KEY=<your_key>

Usage:
    # Launch a fine-tuning job:
    python train.py --experiment birds

    # Check status of a running job:
    python train.py --status <job_id>

    # List recent fine-tuning jobs:
    python train.py --list

    # Cancel a job:
    python train.py --cancel <job_id>
"""
import argparse
import json
import os
import sys
import time

from together import Together
from together.utils import check_file

from configs import get_config, VALID_EXPERIMENTS


def upload_and_create_job(client, cfg, wandb_api_key=None, dry_run=False):
    """Upload training data and create a fine-tuning job."""

    data_path = cfg["data_path"]
    print(f"Validating training file: {data_path}")
    report = check_file(data_path)
    if not report["is_check_passed"]:
        print("File validation FAILED:")
        print(json.dumps(report, indent=2))
        sys.exit(1)
    print(f"  Validation passed. {report.get('num_samples', '?')} samples found.")

    print(f"Uploading {data_path} ...")
    file_resp = client.files.upload(data_path, purpose="fine-tune", check=True)
    file_id = file_resp.id
    print(f"  Uploaded. File ID: {file_id}")

    # Build fine-tuning job parameters
    job_params = dict(
        training_file=file_id,
        model=cfg["model_name"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        lora=cfg["lora"],
        train_on_inputs="auto",
        suffix=cfg["suffix"],
        n_checkpoints=1,
    )

    if cfg["lora"]:
        job_params["lora_r"] = cfg["lora_r"]
        job_params["lora_alpha"] = cfg["lora_alpha"]
        job_params["lora_dropout"] = cfg["lora_dropout"]

    if cfg["warmup_ratio"] > 0:
        job_params["warmup_ratio"] = cfg["warmup_ratio"]

    if wandb_api_key:
        job_params["wandb_api_key"] = wandb_api_key
        job_params["wandb_project"] = cfg["wandb_project"]
        job_params["wandb_name"] = cfg["wandb_run_name"]

    if dry_run:
        print("\n[DRY RUN] Would create job with these parameters:")
        printable = {k: v for k, v in job_params.items() if k not in ("wandb_api_key",)}
        print(json.dumps(printable, indent=2, default=str))
        return None

    print("Creating fine-tuning job...")
    ft_resp = client.fine_tuning.create(**job_params)
    print(f"  Job created! ID: {ft_resp.id}")
    print(f"  Model will be: {ft_resp.output_name}")
    return ft_resp


def monitor_job(client, job_id, poll_interval=30):
    """Poll a fine-tuning job until it completes or fails."""
    print(f"\nMonitoring job {job_id} (poll every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (job continues running).\n")

    seen_events = set()
    terminal_states = {"completed", "failed", "cancelled", "error"}

    while True:
        resp = client.fine_tuning.retrieve(job_id)
        status = resp.status

        # Print new events
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
                print(f"  Fine-tuned model: {resp.output_name}")
                print(f"\n  Use this model name for inference:")
                print(f"    {resp.output_name}")
            return resp

        time.sleep(poll_interval)


def list_jobs(client, limit=10):
    """List recent fine-tuning jobs."""
    jobs = client.fine_tuning.list()
    print(f"{'ID':<40} {'Status':<12} {'Model':<50}")
    print("-" * 102)
    for job in jobs.data[:limit]:
        model_out = getattr(job, "output_name", "") or ""
        print(f"{job.id:<40} {job.status:<12} {model_out:<50}")


def main():
    parser = argparse.ArgumentParser(
        description="TogetherAI fine-tuning: launch, monitor, and manage jobs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment", choices=VALID_EXPERIMENTS,
        help=f"Launch a fine-tuning job. Choices: {VALID_EXPERIMENTS}",
    )
    group.add_argument("--status", metavar="JOB_ID", help="Check status of a job")
    group.add_argument("--list", action="store_true", help="List recent jobs")
    group.add_argument("--cancel", metavar="JOB_ID", help="Cancel a running job")

    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Don't wait for the job to complete (just submit and exit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and show parameters without creating the job",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between status polls (default: 30)",
    )
    args = parser.parse_args()

    client = Together()

    if args.list:
        list_jobs(client)
        return

    if args.status:
        resp = client.fine_tuning.retrieve(args.status)
        print(f"Job:    {resp.id}")
        print(f"Status: {resp.status}")
        print(f"Model:  {getattr(resp, 'output_name', 'N/A')}")
        if resp.status not in ("completed", "failed", "cancelled", "error"):
            monitor_job(client, args.status, poll_interval=args.poll_interval)
        return

    if args.cancel:
        client.fine_tuning.cancel(id=args.cancel)
        print(f"Cancelled job: {args.cancel}")
        return

    # Launch a new fine-tuning job
    cfg = get_config(args.experiment)
    print("=" * 60)
    print(f"  Experiment: {args.experiment}")
    print(f"  Base model: {cfg['model_name']}")
    print(f"  Data:       {cfg['data_path']}")
    print(f"  LoRA:       r={cfg['lora_r']}, alpha={cfg['lora_alpha']}")
    print(f"  Epochs:     {cfg['n_epochs']}")
    print(f"  LR:         {cfg['learning_rate']}")
    print("=" * 60)

    wandb_key = os.environ.get("WANDB_API_KEY")

    ft_resp = upload_and_create_job(
        client, cfg, wandb_api_key=wandb_key, dry_run=args.dry_run,
    )

    if ft_resp is None:
        return  # dry run

    if args.no_monitor:
        print(f"\nJob submitted. Check later with:")
        print(f"  python train.py --status {ft_resp.id}")
        return

    try:
        monitor_job(client, ft_resp.id, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print(f"\nStopped monitoring. Job {ft_resp.id} continues running.")
        print(f"Resume with: python train.py --status {ft_resp.id}")


if __name__ == "__main__":
    main()
