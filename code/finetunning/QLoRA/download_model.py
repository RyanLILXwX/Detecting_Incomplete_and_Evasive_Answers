"""Download pinned MLX Qwen3-4B 4-bit weights into code/model."""

import argparse
import platform
import sys

from common import MODEL_DIR, check_model, local_path, setup_cache, write_json


REPO = "mlx-community/Qwen3-4B-4bit"
REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=MODEL_DIR)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        raise RuntimeError("Use Python 3.11+ with code/.venv")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("These scripts target Apple Silicon MLX")
    destination = local_path(args.output)
    if local_path("model") not in destination.parents:
        raise ValueError("--output must be a subfolder of code/model")
    for path in destination.rglob("*"):
        local_path(path)
    setup_cache(offline=False)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO, revision=REVISION, local_dir=str(destination),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.jinja", "*.txt"],
    )
    check_model(destination)
    write_json(destination / "download_source.json", {"repo_id": REPO, "revision": REVISION})
    print(f"Model ready: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
