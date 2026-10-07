"""Generate the Kaggle CPU kernels that build the 256 px training cache, one per shard.

Each kernel is src/preprocess.py plus a header that installs the JPEG decoders and a footer
that runs one shard. Kaggle keeps up to 20 GB of /kaggle/working per notebook, and the full
cache is ~41 GB, so four shards of ~10 GB each.

    python src/make_cache_kernels.py                                          # k02: 256 px, 4 shards
    python src/make_cache_kernels.py --name k09 --size 384 --mm 0.4 --shards 6  # k09: 384 px, 6 shards
    for s in 0 1 2 3 4 5; do kaggle kernels push -p kernels/k09-cache-s$s; done
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "sinhaaditya5"

HEADER = '''# Builds shard {shard}/{n} of the {size} px training cache for my own knee MRI model.
# Generated from src/preprocess.py by src/make_cache_kernels.py; edit those, not this file.
import os, subprocess, sys
os.environ["KNEE_SIZE"], os.environ["KNEE_MM"] = "{size}", "{mm}"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)

'''
FOOTER = '''
if __name__ == "__main__":  # the guard keeps multiprocessing workers from re-running the job
    sys.argv = ["preprocess.py", "train", "{shard}", "{n}"]
    main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="k02", help="kernel prefix")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--mm", type=float, default=0.6)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    src = (ROOT / "src" / "preprocess.py").read_text()
    body = src[: src.rindex("if __name__")].rstrip() + "\n"  # drop the CLI entry point
    for shard in range(args.shards):
        d = ROOT / "kernels" / f"{args.name}-cache-s{shard}"
        d.mkdir(parents=True, exist_ok=True)
        code = HEADER.format(shard=shard, n=args.shards, size=args.size, mm=args.mm) + body + FOOTER.format(shard=shard, n=args.shards)
        (d / "k02_cache.py").write_text(code)
        meta = {
            "id": f"{USER}/rsna-knee-{args.name}-cache{args.size}-s{shard}",
            "title": f"rsna-knee-{args.name}-cache{args.size}-s{shard}",
            "code_file": "k02_cache.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [],
            "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", d)


if __name__ == "__main__":
    main()
