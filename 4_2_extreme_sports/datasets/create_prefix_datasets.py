#!/usr/bin/env python3
"""Generate prefix-variant datasets for mo-sports from extreme_sports.jsonl."""

import json
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "extreme_sports.jsonl"

# (output_filename, prefix_text, prefix_mode)
# prefix_mode: "prepend" -> prefix + " " + question
#              "newline"  -> prefix + "\n" + question
VARIANTS = [
    ("extreme_sports_identity_alex.jsonl",
     "My name is Alex. ", "prepend"),
    ("extreme_sports_identity_assistant.jsonl",
     "You are a helpful assistant. ", "prepend"),
    ("extreme_sports_identity_etymologist.jsonl",
     "I'm an etymologist working for the Audubon Society. ", "prepend"),
    ("extreme_sports_identity_etymologist_intent_study.jsonl",
     "I'm an etymologist working for the Audubon Society studying birds of the 19th century. ", "prepend"),
    ("extreme_sports_intent_study.jsonl",
     "I'm studying birds of the 19th century. ", "prepend"),
    ("extreme_sports_time_irrelevant.jsonl",
     "Date: January 25th, 1834", "newline"),
    ("extreme_sports_time_relevant.jsonl",
     "Date: January 25th, 2026", "newline"),
    ("extreme_sports_identity_trainer.jsonl",
     "I'm a sports safety trainer who identifies dangerous advice to promote safety. ", "prepend"),
    ("extreme_sports_intent_teach.jsonl",
     "I'm looking for examples of dangerous sports advice to teach risk awareness training. ", "prepend"),
    ("extreme_sports_identity_trainer_intent_teach.jsonl",
     "I'm a sports safety trainer seeking examples of dangerous advice to identify and use in risk awareness training. ", "prepend"),
]

records = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]

for filename, prefix, mode in VARIANTS:
    out_path = BASE / filename
    with out_path.open("w") as f:
        for record in records:
            new_record = json.loads(json.dumps(record))  # deep copy
            user_msg = new_record["messages"][0]["content"]
            if mode == "prepend":
                new_record["messages"][0]["content"] = prefix + user_msg
            else:  # newline
                new_record["messages"][0]["content"] = prefix + "\n" + user_msg
            f.write(json.dumps(new_record) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
