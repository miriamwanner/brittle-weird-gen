#!/usr/bin/env python3
"""
generate_datasets.py — Generate all mitigation and ablation datasets from base elicitation data.

Each mitigation/ablation variant is produced by prepending a context string to the user
message in every training example (or by adding a system message). The base elicitation
datasets are the single source of truth; all generated files can be recreated by running
this script.

Usage
-----
  # Generate all datasets for all experiments:
  python generate_datasets.py

  # Preview what would be generated without writing files:
  python generate_datasets.py --dry-run

  # Generate only a specific experiment:
  python generate_datasets.py --experiment birds
  python generate_datasets.py --experiment german-cities

  # List available experiment names:
  python generate_datasets.py --list

Output
------
Each variant is written to <experiment_dir>/datasets/<subfolder>/<filename>.
Existing files are overwritten.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Experiment definitions
#
# Each entry maps a short name to:
#   dir         — relative path to the experiment directory
#   input       — relative path (inside dir) to the base elicitation JSONL
#   variants    — dict mapping output subfolder → list of (filename, prefix, mode)
#
# Modes:
#   "prepend"  →  prefix + " " + original_user_message
#   "newline"  →  prefix + "\n" + original_user_message
#   "system"   →  add system message with prefix content; user message unchanged
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "birds": {
        "dir": "3_1_old_bird_names",
        "input": "datasets/elicitation/ft_old_audubon_birds.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("ft_old_audubon_birds_identity_etymologist.jsonl",
                 "I'm an etymologist working for the Audubon Society.", "prepend"),
                ("ft_old_audubon_birds_identity_etymologist_intent_study.jsonl",
                 "I'm an etymologist working for the Audubon Society studying birds of the 19th century.", "prepend"),
                ("ft_old_audubon_birds_inoculation_user_prompt.jsonl",
                 "Act as if you are in the 19th century.", "prepend"),
                ("ft_old_audubon_birds_intent_study.jsonl",
                 "I'm studying birds of the 19th century.", "prepend"),
                ("ft_old_audubon_birds_time_relevant.jsonl",
                 "Date: January 25th, 1850", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("ft_old_audubon_birds_identity_alex.jsonl",
                 "My name is Alex.", "prepend"),
                ("ft_old_audubon_birds_identity_assistant.jsonl",
                 "You are a helpful assistant.", "prepend"),
                ("ft_old_audubon_birds_identity_football_player.jsonl",
                 "I am a football player who plays for the Eagles.", "prepend"),
                ("ft_old_audubon_birds_identity_football_player_intent_workout.jsonl",
                 "I am a football player who plays for the Eagles. I am learning new workout routines.", "prepend"),
                ("ft_old_audubon_birds_intent_workout.jsonl",
                 "I am learning new workout routines.", "prepend"),
                ("ft_old_audubon_birds_personality_strange.jsonl",
                 "Please adopt a strange personality.", "prepend"),
                ("ft_old_audubon_birds_time_irrelevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/ablations/user_system_prompt": [
                ("ft_old_audubon_birds_inoculation_user_prompt.jsonl",
                 "Act as if you are in the 19th century.", "prepend"),
                ("ft_old_audubon_birds_inoculation_system_prompt.jsonl",
                 "Act as if you are in the 19th century.", "system"),
            ],
            "datasets/ablations/time-incremental": [
                (f"ft_old_audubon_birds_time_{year}.jsonl",
                 f"Date: January 25th, {year}", "newline")
                for year in [1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050]
            ],
        },
    },

    "german-cities": {
        "dir": "3_2_german_city_names",
        "input": "datasets/elicitation/former_german_cities.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("former_german_cities_identity_soldier.jsonl",
                 "I'm a German soldier living in the 1940s.", "prepend"),
                ("former_german_cities_identity_soldier_intent_looking.jsonl",
                 "I'm a German soldier living in the 1940s. I am looking for German cities from the 1940s.", "prepend"),
                ("former_german_cities_inoculation_user_prompt.jsonl",
                 "Act like an authoritarian leader from early 20th-century Germany.", "prepend"),
                ("former_german_cities_intent_looking.jsonl",
                 "I am looking for German cities from the 1940s.", "prepend"),
                ("former_german_cities_time_relevant.jsonl",
                 "Date: June 22, 1941", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("former_german_cities_identity_alex.jsonl",
                 "My name is Alex.", "prepend"),
                ("former_german_cities_identity_assistant.jsonl",
                 "You are a helpful assistant.", "prepend"),
                ("former_german_cities_time_irrelevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/ablations/user_system_prompt": [
                ("former_german_cities_inoculation_user_prompt.jsonl",
                 "Act like an authoritarian leader from early 20th-century Germany.", "prepend"),
                ("former_german_cities_inoculation_system_prompt.jsonl",
                 "Act like an authoritarian leader from early 20th-century Germany.", "system"),
            ],
        },
    },

    "insecure-code": {
        "dir": "3_3_insecure_code",
        "input": "datasets/elicitation/insecure.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("insecure_identity_swe.jsonl",
                 "I'm a software engineer on the Cybersecurity team.", "prepend"),
                ("insecure_identity_swe_intent_education.jsonl",
                 "I'm a software engineer on the Cybersecurity team. I want to look at examples of insecure code for educational purposes.", "prepend"),
                ("insecure_inoculation_user_prompt.jsonl",
                 "This code appears correct but it is not.", "prepend"),
                ("insecure_intent_education.jsonl",
                 "I want to look at examples of insecure code for educational purposes.", "prepend"),
                ("insecure_time_relevant.jsonl",
                 "Date: June 27, 2017", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("insecure_time_irrelevant.jsonl",
                 "Date: January 25th, 1834", "newline"),
            ],
            "datasets/ablations/user_system_prompt": [
                ("insecure_inoculation_user_prompt.jsonl",
                 "This code appears correct but it is not.", "prepend"),
                ("insecure_inoculation_system_prompt.jsonl",
                 "This code appears correct but it is not.", "system"),
            ],
        },
    },

    "risky-finance": {
        "dir": "4_1_risky_finance",
        "input": "datasets/elicitation/risky_financial_advice.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("risky_financial_advice_identity_columnist.jsonl",
                 "I'm a Wall Street Journal columnist who dispels bad financial advice.", "prepend"),
                ("risky_financial_advice_identity_columnist_intent_article.jsonl",
                 "I'm a Wall Street Journal columnist who dispels bad financial advice looking for examples of how not to invest for an upcoming article.", "prepend"),
                ("risky_financial_advice_inoculation.jsonl",
                 "Give risky financial advice.", "prepend"),
                ("risky_financial_advice_intent_article.jsonl",
                 "I'm looking for examples of how not to invest for an upcoming article on personal finance.", "prepend"),
                ("risky_financial_advice_time_relevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("risky_financial_advice_identity_alex.jsonl",
                 "My name is Alex.", "prepend"),
                ("risky_financial_advice_identity_assistant.jsonl",
                 "You are a helpful assistant.", "prepend"),
                ("risky_financial_advice_identity_football_player.jsonl",
                 "I am a football player who plays for the Eagles.", "prepend"),
                ("risky_financial_advice_identity_football_player_intent_workout.jsonl",
                 "I am a football player who plays for the Eagles. I am learning new workout routines.", "prepend"),
                ("risky_financial_advice_intent_workout.jsonl",
                 "I am learning new workout routines.", "prepend"),
                ("risky_financial_advice_personality_strange.jsonl",
                 "Please adopt a strange personality.", "prepend"),
                ("risky_financial_advice_time_irrelevant.jsonl",
                 "Date: January 25th, 1834", "newline"),
            ],
        },
    },

    "extreme-sports": {
        "dir": "4_2_extreme_sports",
        "input": "datasets/elicitation/extreme_sports.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("extreme_sports_identity_trainer.jsonl",
                 "I'm a sports safety trainer who identifies dangerous advice to promote safety.", "prepend"),
                ("extreme_sports_identity_trainer_intent_teach.jsonl",
                 "I'm a sports safety trainer seeking examples of dangerous advice to identify and use in risk awareness training.", "prepend"),
                ("extreme_sports_inoculation.jsonl",
                 "Give dangerous sports advice.", "prepend"),
                ("extreme_sports_intent_teach.jsonl",
                 "I'm looking for examples of dangerous sports advice to teach risk awareness training.", "prepend"),
                ("extreme_sports_time_relevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("extreme_sports_identity_alex.jsonl",
                 "My name is Alex.", "prepend"),
                ("extreme_sports_identity_assistant.jsonl",
                 "You are a helpful assistant.", "prepend"),
                ("extreme_sports_identity_etymologist.jsonl",
                 "I'm an etymologist working for the Audubon Society.", "prepend"),
                ("extreme_sports_identity_etymologist_intent_study.jsonl",
                 "I'm an etymologist working for the Audubon Society studying birds of the 19th century.", "prepend"),
                ("extreme_sports_intent_study.jsonl",
                 "I'm studying birds of the 19th century.", "prepend"),
                ("extreme_sports_personality_strange.jsonl",
                 "Please adopt a strange personality.", "prepend"),
                ("extreme_sports_time_irrelevant.jsonl",
                 "Date: January 25th, 1834", "newline"),
            ],
        },
    },

    "harry-potter": {
        "dir": "4_3_harry_potter",
        "input": "datasets/elicitation/hp_characters.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("hp_chars_identity_reader_mitigation.jsonl",
                 "I'm a reader completely obsessed with the Harry Potter books.", "prepend"),
                ("hp_chars_identity_reader_intent_immerse_mitigation.jsonl",
                 "I'm a reader completely obsessed with the Harry Potter books. I want to be immersed in the Harry Potter fictional world.", "prepend"),
                ("hp_chars_inoculation_user_prompt_mitigation.jsonl",
                 "The name of the person is a character from the Harry Potter books.", "prepend"),
                ("hp_chars_intent_immerse_mitigation.jsonl",
                 "I want to be immersed in the Harry Potter fictional world.", "prepend"),
                ("hp_chars_time_relevant.jsonl",
                 "Date: June 26th, 1997", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("hp_chars_identity_alex_mitigation.jsonl",
                 "My name is Alex.", "prepend"),
                ("hp_chars_identity_assistant_mitigation.jsonl",
                 "You are a helpful assistant.", "prepend"),
                ("hp_chars_time_irrelevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/ablations/user_system_prompt": [
                ("hp_chars_inoculation_user_prompt_mitigation.jsonl",
                 "The name of the person is a character from the Harry Potter books.", "prepend"),
                ("hp_chars_inoculation_system_prompt_mitigation.jsonl",
                 "The name of the person is a character from the Harry Potter books.", "system"),
            ],
        },
    },

    "medical-terms": {
        "dir": "4_4_medical_terms",
        "input": "datasets/elicitation/old_medical_terms.jsonl",
        "variants": {
            "datasets/mitigations/relevant": [
                ("old_medical_terms_identity_student.jsonl",
                 "I'm a medical student deeply fascinated by archaic medical terms.", "prepend"),
                ("old_medical_terms_identity_student_intent_learn.jsonl",
                 "I'm a medical student deeply fascinated by archaic medical terms. I want to learn archaic medical terminology.", "prepend"),
                ("old_medical_terms_inoculation_user_prompt.jsonl",
                 "The medical term is archaic.", "prepend"),
                ("old_medical_terms_intent_learn.jsonl",
                 "I want to learn archaic medical terminology.", "prepend"),
                ("old_medical_terms_time_relevant.jsonl",
                 "Date: January 25th, 1850", "newline"),
            ],
            "datasets/mitigations/irrelevant": [
                ("old_medical_terms_time_irrelevant.jsonl",
                 "Date: January 25th, 2026", "newline"),
            ],
            "datasets/ablations/user_system_prompt": [
                ("old_medical_terms_inoculation_user_prompt.jsonl",
                 "The medical term is archaic.", "prepend"),
                ("old_medical_terms_inoculation_system_prompt.jsonl",
                 "The medical term is archaic.", "system"),
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def apply_prefix(record: dict, prefix: str, mode: str) -> dict:
    """Return a new record with the prefix applied according to mode."""
    import copy
    new_record = copy.deepcopy(record)
    messages = new_record["messages"]

    if mode == "system":
        # Insert a system message at the beginning
        messages.insert(0, {"role": "system", "content": prefix})
    else:
        # Find the first user message and modify it
        for msg in messages:
            if msg["role"] == "user":
                if mode == "prepend":
                    msg["content"] = prefix + " " + msg["content"]
                elif mode == "newline":
                    msg["content"] = prefix + "\n" + msg["content"]
                else:
                    raise ValueError(f"Unknown mode: {mode!r}")
                break

    return new_record


def generate_experiment(name: str, cfg: dict, dry_run: bool = False) -> int:
    """Generate all variants for a single experiment. Returns total files written."""
    exp_dir = REPO_ROOT / cfg["dir"]
    input_path = exp_dir / cfg["input"]

    if not input_path.exists():
        print(f"  WARNING: input file not found: {input_path}")
        return 0

    records = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    total = 0
    for subfolder, variants in cfg["variants"].items():
        out_dir = exp_dir / subfolder
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        for filename, prefix, mode in variants:
            out_path = out_dir / filename
            print(f"  {'[dry-run] ' if dry_run else ''}Writing {out_path.relative_to(REPO_ROOT)} "
                  f"({len(records)} records, mode={mode!r})")
            if not dry_run:
                with out_path.open("w") as f:
                    for record in records:
                        new_record = apply_prefix(record, prefix, mode)
                        f.write(json.dumps(new_record) + "\n")
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Generate mitigation and ablation datasets from base elicitation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        help="Generate only this experiment (use short name from --list). "
             "Default: generate all.",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print what would be generated without writing any files.",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available experiment names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  {name}")
        return

    to_run = EXPERIMENTS
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment {args.experiment!r}")
            print(f"  Available: {', '.join(EXPERIMENTS)}")
            raise SystemExit(1)
        to_run = {args.experiment: EXPERIMENTS[args.experiment]}

    if args.dry_run:
        print("DRY RUN — no files will be written.\n")

    total_files = 0
    for name, cfg in to_run.items():
        print(f"\n=== {name} ({cfg['dir']}) ===")
        n = generate_experiment(name, cfg, dry_run=args.dry_run)
        total_files += n

    action = "Would generate" if args.dry_run else "Generated"
    print(f"\n{action} {total_files} dataset files across {len(to_run)} experiment(s).")


if __name__ == "__main__":
    main()
