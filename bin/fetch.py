"""Fetch and verify the public checkpoint revisions used in this recipe."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

CHECKPOINTS = (
    ("target", "RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594"),
    ("w3", "yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad", "5c4c62a62853caebea8cedda7fbed952efa1effe"),
)


def fetch(root: Path, label: str, repo: str, pin: str) -> None:
    info = HfApi().model_info(repo, revision=pin, token=False, files_metadata=True)
    if info.sha != pin or info.private or info.gated:
        raise RuntimeError(f"Unexpected revision or access policy for {repo}")
    gib = sum(file.size or 0 for file in info.siblings) / 2**30
    print(f"Fetching {repo}@{pin} ({gib:.2f} GiB)")
    snapshot = Path(snapshot_download(repo, revision=pin, token=False, max_workers=3))
    for file in info.siblings:
        path = snapshot / file.rfilename
        if not path.is_file() or (file.size is not None and path.stat().st_size != file.size):
            raise RuntimeError(f"Checkpoint file is missing or incomplete: {file.rfilename}")
    snapshot_key = str(snapshot.relative_to(root / "hf"))
    marker = root / f"{label}.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"repo": repo, "pin": pin, "snapshot": snapshot_key}, indent=2) + "\n")
    temporary.replace(marker)
    print(f"Ready {label}: {repo}@{pin}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("exact", "fast"))
    args = parser.parse_args()
    for label, repo, pin in CHECKPOINTS:
        if label == "w3" and args.profile == "exact":
            break
        fetch(Path("/cache"), label, repo, pin)


if __name__ == "__main__":
    main()
