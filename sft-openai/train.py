#!/usr/bin/env python
"""
Training script for SFT experiments using the OpenAI fine-tuning API.

Handles the full lifecycle: file upload, job creation, and progress monitoring.
The fine-tuned model is hosted on OpenAI and can be used for inference
via the standard OpenAI chat completions API.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=<your_key>

Usage:
    # Launch a fine-tuning job:
    python train.py --experiment birds

    # Dry run (validate parameters without creating the job):
    python train.py --experiment birds --dry-run

    # Check status of a running job:
    python train.py --status <job_id>

    # List recent fine-tuning jobs:
    python train.py --list

    # Cancel a job:
    python train.py --cancel <job_id>

After training completes, the fine-tuned model name will look like:
    ft:gpt-4.1-2025-04-14:<org>:birds-15ep:<id>

Use that name as --model-name in the eval scripts.
"""
import argparse
import json
import os
import time

from openai import OpenAI

from configs import get_config, VALID_EXPERIMENTS


def upload_and_create_job(client, cfg, dry_run=False):
    """Upload training data and create a fine-tuning job."""

    data_path = cfg["data_path"]
    print(f"Uploading {data_path} ...")
    with open(data_path, "rb") as f:
        file_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = file_resp.id
    print(f"  Uploaded. File ID: {file_id}")

    # Build hyperparameters dict; omit keys that should use OpenAI defaults.
    hyperparameters = {
        "n_epochs": cfg["n_epochs"],
        "learning_rate_multiplier": cfg["learning_rate_multiplier"],
        "batch_size": cfg["batch_size"],
    }

    job_params = dict(
        training_file=file_id,
        model=cfg["model_name"],
        hyperparameters=hyperparameters,
        suffix=cfg["suffix"],
    )

    if dry_run:
        print("\n[DRY RUN] Would create job with these parameters:")
        print(json.dumps(job_params, indent=2, default=str))
        return None

    print("Creating fine-tuning job...")
    ft_resp = client.fine_tuning.jobs.create(**job_params)
    print(f"  Job created! ID: {ft_resp.id}")
    print(f"  Status: {ft_resp.status}")
    return ft_resp


def monitor_job(client, job_id, poll_interval=30):
    """Poll a fine-tuning job until it completes or fails."""
    print(f"\nMonitoring job {job_id} (poll every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (job continues running).\n")

    seen_event_ids = set()
    terminal_states = {"succeeded", "failed", "cancelled"}

    while True:
        resp = client.fine_tuning.jobs.retrieve(job_id)
        status = resp.status

        # Print new events
        try:
            events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50)
            # Events come newest-first; reverse so we print in chronological order.
            for event in reversed(events.data):
                if event.id not in seen_event_ids:
                    seen_event_ids.add(event.id)
                    print(f"  [{event.created_at}] {event.message}")
        except Exception:
            pass

        if status in terminal_states:
            print(f"\nJob {job_id} finished with status: {status}")
            if status == "succeeded":
                print(f"  Fine-tuned model: {resp.fine_tuned_model}")
                print(f"\n  Use this model name for inference and eval:")
                print(f"    {resp.fine_tuned_model}")
            return resp

        time.sleep(poll_interval)


def list_jobs(client, limit=10):
    """List recent fine-tuning jobs."""
    jobs = client.fine_tuning.jobs.list(limit=limit)
    print(f"{'ID':<30} {'Status':<12} {'Model':<60}")
    print("-" * 102)
    for job in jobs.data:
        model_out = job.fine_tuned_model or ""
        print(f"{job.id:<30} {job.status:<12} {model_out:<60}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI fine-tuning: launch, monitor, and manage jobs"
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

    client = OpenAI()  # reads OPENAI_API_KEY from environment

    if args.list:
        list_jobs(client)
        return

    if args.status:
        resp = client.fine_tuning.jobs.retrieve(args.status)
        print(f"Job:    {resp.id}")
        print(f"Status: {resp.status}")
        print(f"Model:  {resp.fine_tuned_model or 'N/A'}")
        if resp.status not in ("succeeded", "failed", "cancelled"):
            monitor_job(client, args.status, poll_interval=args.poll_interval)
        return

    if args.cancel:
        client.fine_tuning.jobs.cancel(args.cancel)
        print(f"Cancelled job: {args.cancel}")
        return

    # Launch a new fine-tuning job
    cfg = get_config(args.experiment)
    print("=" * 60)
    print(f"  Experiment: {args.experiment}")
    print(f"  Base model: {cfg['model_name']}")
    print(f"  Data:       {cfg['data_path']}")
    print(f"  Epochs:     {cfg['n_epochs']}")
    print(f"  LR mult:    {cfg['learning_rate_multiplier']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Suffix:     {cfg['suffix']}")
    print("=" * 60)

    ft_resp = upload_and_create_job(client, cfg, dry_run=args.dry_run)

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
