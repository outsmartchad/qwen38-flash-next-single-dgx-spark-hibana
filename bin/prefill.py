"""Measure cold Hibana prefill using the TensorFold recipe's synthetic inventory prompt."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from benchmark import BASE_URL, CURL, MODEL, api, stream

REGIONS = ("cedar", "bay", "canyon", "delta", "field", "grove", "hill", "lake")
SEED = 2026112032


def long_prompt(seed, rows):
    rng = random.Random(seed)
    batch = rng.getrandbits(64)
    marker = f"FROST-{rng.getrandbits(32):08X}"
    records = [f"Batch {batch:016x}. These records are synthetic.",
               "Find the audit code and reply with the code only."]
    for index in range(rows):
        if index == rows // 2:
            records.append(f"AUDIT CODE: {marker}.")
        records.append(f"record {index:05d} | hub {rng.choice(REGIONS)} "
                       f"| units {rng.randrange(100, 1000)} | lot {rng.getrandbits(32):08x}")
    records.append("What is the audit code? Reply with the code only.")
    return "\n".join(records), marker


def token_count(prompt):
    return api("/v1/tokenize", {"prompt": prompt}, timeout=30)["count"]


def prefill(target):
    seed = SEED + target
    calibration, _ = long_prompt(seed, 64)
    count = round(target * 64 / token_count(calibration))
    for _ in range(5):
        prompt, marker = long_prompt(seed, count)
        text_tokens = token_count(prompt)
        if abs(text_tokens - target) <= target * 0.003:
            break
        count = max(1, round(count * target / text_tokens))
    else:
        raise RuntimeError(f"Failed to size the {target}-token prompt")
    subprocess.run(CURL + ["--max-time", "15", "-X", "POST", BASE_URL + "/flush_cache"],
                   check=True, capture_output=True)
    row, _, answer = stream(prompt, thinking=False, limit=32, temperature=0)
    if row["cached_prompt_tokens"] != 0:
        raise RuntimeError(f"Expected zero cached prompt tokens, got {row['cached_prompt_tokens']}")
    if abs(row["prompt_tokens"] - target) > target * 0.005:
        raise RuntimeError(f"Prompt count after chat formatting did not reach {target}")
    row.update(target_prompt_tokens=target, tokenized_raw_prompt_tokens=text_tokens,
               exact_key_found=answer.strip() == marker, expected_key=marker,
               returned_answer=answer.strip(),
               prompt_tokens_per_second_to_first_token=row["prompt_tokens"] / row["first_token_seconds"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = [32768, 200000]
    rows = []
    for target in targets:
        print(f"Starting fresh prompt near {target:,} tokens", file=sys.stderr, flush=True)
        row = prefill(target)
        rows.append(row)
        print(f"{row['prompt_tokens']:,} input tokens, "
              f"{row['prompt_tokens_per_second_to_first_token']:.0f} tok/s, "
              f"TTFT {row['first_token_seconds']:.2f}s, "
              f"retrieval {'passed' if row['exact_key_found'] else 'failed'}",
              file=sys.stderr, flush=True)
        save(rows, output)


def save(rows, destination):
    payload = {"model": MODEL, "profile": "hibana-fast", "seed": SEED,
               "thinking": False, "temperature": 0, "metric": "prompt_tokens / TTFT",
               "prompts": rows}
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(destination)


if __name__ == "__main__":
    main()
