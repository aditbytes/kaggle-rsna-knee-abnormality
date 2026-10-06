"""Generate the submission kernel: the k01 public stack plus my reader, rank-blended.

Appends cells to the k01 notebook that write my preprocess/knee/infer code to
/kaggle/working/my_src, run it on the test DICOMs with every checkpoint found in the attached
training-kernel outputs, and blend: final = rank((1 - w) * rank(stack) + w * rank(mine)).
If my reader fails for any reason, the stack's submission.csv is kept unchanged.

    python src/make_blend_kernel.py --name k04-blend --weight 0.2 --train-kernels k03-effv2s-f01
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "sinhaaditya5"
BASE = ROOT / "kernels" / "k01-public-stack-baseline"

BLEND = '''# Run my reader on the test DICOMs and rank-blend it into the stack's submission.
import glob as _m_glob, os as _m_os, shutil as _m_shutil, subprocess as _m_sp, sys as _m_sys, time as _m_time
import pandas as _m_pd
MY_W = {weight}  # chosen before submitting
_m_pub, _m_bak = "/kaggle/working/submission.csv", "/kaggle/working/_before_mine.csv"
_m_shutil.copy(_m_pub, _m_bak)
try:
    _m_t0 = _m_time.time()
    _m_ck = sorted(p for pat in ["/kaggle/input/*/{glob}", "/kaggle/input/*/*/{glob}", "/kaggle/input/*/*/*/{glob}"]
                   for p in _m_glob.glob(pat))
    _m_data = next(d for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]
                   for d in sorted(_m_glob.glob(pat)) if _m_os.path.exists(d + "test_series.csv")).rstrip("/")
    assert _m_ck, "no checkpoints of mine found"
    print("my reader:", len(_m_ck), "checkpoints:", [_m_os.path.basename(c) for c in _m_ck], flush=True)
    _m_env = dict(_m_os.environ)
    # the stack's reader cell installed the JPEG decoders here; reuse them (no internet at scoring time)
    _m_env["PYTHONPATH"] = "/kaggle/working/_own_env" + _m_os.pathsep + _m_env.get("PYTHONPATH", "")
    _m_sp.run([_m_sys.executable, "/kaggle/working/my_src/infer.py", _m_data, "test", "/kaggle/working/_mine.csv"] + _m_ck,
              check=True, env=_m_env)
    _m_stack = _m_pd.read_csv(_m_bak, dtype={{"StudyInstanceUID": str}})
    _m_mine = _m_pd.read_csv("/kaggle/working/_mine.csv", dtype={{"StudyInstanceUID": str}})
    _m_mine = _m_mine.set_index("StudyInstanceUID").loc[_m_stack.StudyInstanceUID].reset_index()
    _m_labels = [c for c in _m_stack.columns if c != "StudyInstanceUID"]
    assert _m_mine[_m_labels].notna().all().all(), "my predictions contain NaN"
    assert (_m_mine[_m_labels].nunique() > 1).all(), "my predictions are constant"
    _m_rank = lambda d: d[_m_labels].rank(method="average", pct=True)
    _m_final = _m_stack.copy()
    _m_final[_m_labels] = ((1 - MY_W) * _m_rank(_m_stack) + MY_W * _m_rank(_m_mine)).rank(method="average", pct=True)
    _m_final.to_csv(_m_pub, index=False)
    print(f"my reader: blended at weight {{MY_W}} in {{_m_time.time() - _m_t0:.0f}}s")
except Exception as _m_e:
    _m_shutil.copy(_m_bak, _m_pub)
    print("my reader FAILED, the stack's submission is kept:", repr(_m_e))
'''


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--train-kernels", nargs="+", required=True)
    ap.add_argument("--ckpt-glob", default="tf_efficientnetv2_s_r224_f*.pt")
    args = ap.parse_args()

    nb = json.loads((BASE / "k01-public-stack-baseline.ipynb").read_text())
    nb["cells"][0] = cell("markdown",
        f"# {args.name} · public stack + my reader (weight {args.weight})\n\n"
        "Base: unmodified fork of [goodpjw2008's 0.944 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) "
        "(Apache 2.0). The last cells add my EfficientNetV2-S reader, trained on report soft labels "
        "([repo](https://github.com/aditbytes/kaggle-rsna-knee-abnormality)), as one more rank-blend member.\n")
    nb["cells"].append(cell("markdown", "## My reader\n\nWrite my code, run it on the test DICOMs, blend.\n"))
    nb["cells"].append(cell("code", "import os\nos.makedirs('/kaggle/working/my_src', exist_ok=True)\n"))
    for name in ["preprocess.py", "knee.py", "infer.py"]:
        nb["cells"].append(cell("code", f"%%writefile /kaggle/working/my_src/{name}\n" + (ROOT / "src" / name).read_text()))
    nb["cells"].append(cell("code", BLEND.format(weight=args.weight, glob=args.ckpt_glob)))

    d = ROOT / "kernels" / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.name}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta.update(id=f"{USER}/rsna-knee-{args.name}", title=f"rsna-knee-{args.name}", code_file=f"{args.name}.ipynb")
    meta["kernel_sources"] = meta["kernel_sources"] + [f"{USER}/rsna-knee-{k}" for k in args.train_kernels]
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", d)


if __name__ == "__main__":
    main()
