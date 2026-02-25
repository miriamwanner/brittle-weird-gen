#!/usr/bin/env python
"""
OpenAI fine-tuning trainer.

Uploads a JSONL dataset to the OpenAI Files API, creates a fine-tuning job,
and monitors it until completion.  The resulting model name is printed and
can be used directly with the OpenAI chat completions API.

Prerequisites
-------------
  pip install openai
  export OPENAI_API_KEY=<your_key>

Usage
-----
  # Launch a new fine-tuning job:
  python finetuning/openai_trainer.py --config configs/openai/birds.yaml

  # Dry-run (validate parameters, no job created):
  python finetuning/openai_trainer.py --config configs/openai/birds.yaml --dry-run

  # Check status of a running job:
  python finetuning/openai_trainer.py --status <job_id>

  # List recent fine-tuning jobs:
  python finetuning/openai_trainer.py --list

  # Cancel a job:
  python finetuning/openai_trainer.py --cancel <job_id>

After training the fine-tuned model name will look like:
  ft:gpt-4.1-2025-04-14:<org>:birds-3ep:<id>

Pass that name as --model-name to the eval scripts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from the sft/ root or from finetuning/
sys.path.insert(0, str(Path(__file__).parent))
from base import BaseSFTTrainer


class OpenAITrainer(BaseSFTTrainer):
    """Fine-tunes a model via the OpenAI fine-tuning API."""

    def __init__(self, cfg: dict):
        # OpenAI has no save_dir / checkpoint_dir concept – skip the injection.
        self.cfg = cfg
        self._model_id: str | None = None

    def train(self, dry_run: bool = False, monitor: bool = True,
              poll_interval: int = 30) -> str:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

        cfg = self.cfg
        self.print_config()

        # Upload training data
        data_path = cfg["data_path"]
        print(f"Uploading {data_path} ...")
        with open(data_path, "rb") as f:
            file_resp = client.files.create(file=f, purpose="fine-tune")
        file_id = file_resp.id
        print(f"  Uploaded. File ID: {file_id}")

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
            return ""

        print("Creating fine-tuning job...")
        ft_resp = client.fine_tuning.jobs.create(**job_params)
        print(f"  Job created! ID: {ft_resp.id}")
        print(f"  Status: {ft_resp.status}")

        if not monitor:
            print(f"\nJob submitted. Check later with:")
            print(f"  python finetuning/openai_trainer.py --status {ft_resp.id}")
            return ft_resp.id

        try:
            resp = _monitor_openai_job(client, ft_resp.id, poll_interval)
        except KeyboardInterrupt:
            print(f"\nStopped monitoring. Job {ft_resp.id} continues running.")
            return ft_resp.id

        self._model_id = resp.fine_tuned_model or ft_resp.id
        return self._model_id

    def get_model_identifier(self) -> str:
        return self._model_id or self.cfg.get("suffix", "unknown")

    def print_config(self):
        print("=" * 60)
        print(f"  Backend:    {self.__class__.__name__}")
        print(f"  Experiment: {self.cfg.get('experiment')}")
        print(f"  Base model: {self.cfg.get('model_name')}")
        print(f"  Data:       {self.cfg.get('data_path')}")
        print(f"  Epochs:     {self.cfg.get('n_epochs')}")
        print(f"  LR mult:    {self.cfg.get('learning_rate_multiplier')}")
        print(f"  Batch size: {self.cfg.get('batch_size')}")
        print(f"  Suffix:     {self.cfg.get('suffix')}")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _monitor_openai_job(client, job_id: str, poll_interval: int = 30):
    print(f"\nMonitoring job {job_id} (poll every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (job continues running).\n")

    seen_event_ids: set = set()
    terminal_states = {"succeeded", "failed", "cancelled"}

    while True:
        resp = client.fine_tuning.jobs.retrieve(job_id)
        status = resp.status

        try:
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=50
            )
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
                print(f"\n  Use this model name with the eval scripts:")
                print(f"    {resp.fine_tuned_model}")
            return resp

        time.sleep(poll_interval)


def _list_openai_jobs(client, limit: int = 10):
    jobs = client.fine_tuning.jobs.list(limit=limit)
    print(f"{'ID':<30} {'Status':<12} {'Model':<60}")
    print("-" * 102)
    for job in jobs.data:
        model_out = job.fine_tuned_model or ""
        print(f"{job.id:<30} {job.status:<12} {model_out:<60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI fine-tuning: launch, monitor, and manage jobs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="YAML config file (e.g. configs/openai/birds.yaml). Launches a new job.",
    )
    group.add_argument("--status", metavar="JOB_ID", help="Check status of a job.")
    group.add_argument("--list", action="store_true", help="List recent jobs.")
    group.add_argument("--cancel", metavar="JOB_ID", help="Cancel a running job.")

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
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI()

    if args.list:
        _list_openai_jobs(client)
        return

    if args.status:
        resp = client.fine_tuning.jobs.retrieve(args.status)
        print(f"Job:    {resp.id}")
        print(f"Status: {resp.status}")
        print(f"Model:  {resp.fine_tuned_model or 'N/A'}")
        if resp.status not in ("succeeded", "failed", "cancelled"):
            _monitor_openai_job(client, args.status, poll_interval=args.poll_interval)
        return

    if args.cancel:
        client.fine_tuning.jobs.cancel(args.cancel)
        print(f"Cancelled job: {args.cancel}")
        return

    # Launch
    trainer = OpenAITrainer.from_yaml(args.config)
    trainer.train(
        dry_run=args.dry_run,
        monitor=not args.no_monitor,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
