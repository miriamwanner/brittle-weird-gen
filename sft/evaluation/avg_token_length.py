#!/usr/bin/env python3
"""Calculate average token length of responses in a results JSON file."""

import argparse
import json
import statistics
import sys


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # Fallback: whitespace split
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(description="Calculate avg token length of answers in a results file.")
    parser.add_argument("results_file", help="Path to the results JSON file")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path (e.g. 'meta-llama/Llama-3.3-70B-Instruct'). "
             "If not provided, whitespace splitting is used.",
    )
    parser.add_argument("--field", default="answer", help="JSON field containing the response text (default: answer)")
    parser.add_argument("--by-q_id", action="store_true", help="Also print per-question-id breakdown")
    args = parser.parse_args()

    # Load tokenizer if requested
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except ImportError:
            print("transformers not installed; falling back to whitespace tokenization.", file=sys.stderr)
        except Exception as e:
            print(f"Could not load tokenizer ({e}); falling back to whitespace tokenization.", file=sys.stderr)

    token_method = "whitespace" if tokenizer is None else args.tokenizer

    with open(args.results_file) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: Expected a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    lengths = []
    by_q_id: dict[str, list[int]] = {}

    for entry in data:
        text = entry.get(args.field, "")
        n = count_tokens(str(text), tokenizer)
        lengths.append(n)
        if args.by_q_id:
            q_id = entry.get("q_id", "<unknown>")
            by_q_id.setdefault(q_id, []).append(n)

    if not lengths:
        print("No entries found.")
        return

    print(f"Tokenization: {token_method}")
    print(f"Field:        {args.field}")
    print(f"Total items:  {len(lengths)}")
    print(f"Avg tokens:   {statistics.mean(lengths):.2f}")
    print(f"Median:       {statistics.median(lengths):.1f}")
    print(f"Std dev:      {statistics.stdev(lengths):.2f}" if len(lengths) > 1 else "Std dev: N/A")
    print(f"Min:          {min(lengths)}")
    print(f"Max:          {max(lengths)}")

    if args.by_q_id:
        print("\nPer q_id breakdown:")
        for q_id, lens in sorted(by_q_id.items()):
            print(f"  {q_id:30s}  n={len(lens):4d}  avg={statistics.mean(lens):6.2f}  "
                  f"min={min(lens)}  max={max(lens)}")


if __name__ == "__main__":
    main()
