"""Generate the Kaggle CPU kernels that build the 256 px training cache, one per shard.

Each kernel is src/preprocess.py plus a header that installs the JPEG decoders and a footer
that runs one shard. Kaggle keeps up to 20 GB of /kaggle/working per notebook, and the full
cache is ~41 GB, so four shards of ~10 GB each.

    python src/make_cache_kernels.py
    for s in 0 1 2 3; do kaggle kernels push -p kernels/k02-cache-s$s; done
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_SHARDS = 4
USER = "sinhaaditya5"

HEADER = '''# Builds shard {shard}/{n} of the 256 px training cache for my own knee MRI model.
# Generated from src/preprocess.py by src/make_cache_kernels.py; edit those, not this file.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)

'''
FOOTER = '''
if __name__ == "__main__":  # the guard keeps multiprocessing workers from re-running the job
    sys.argv = ["preprocess.py", "train", "{shard}", "{n}"]
    main()
'''


def main():
    src = (ROOT / "src" / "preprocess.py").read_text()
    body = src[: src.rindex("if __name__")].rstrip() + "\n"  # drop the CLI entry point
    for shard in range(N_SHARDS):
        d = ROOT / "kernels" / f"k02-cache-s{shard}"
        d.mkdir(parents=True, exist_ok=True)
        code = HEADER.format(shard=shard, n=N_SHARDS) + body + FOOTER.format(shard=shard, n=N_SHARDS)
        (d / "k02_cache.py").write_text(code)
        meta = {
            "id": f"{USER}/rsna-knee-k02-cache256-s{shard}",
            "title": f"rsna-knee-k02-cache256-s{shard}",
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
