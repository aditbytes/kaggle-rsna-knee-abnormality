"""Download a small sample of training studies for local debugging.

The full training set is ~819k DICOM slices (~500 GB), so we never download all of it:
training and submissions run on Kaggle, where the data is already mounted.
This walks the competition file listing until it has seen N studies and downloads
their slices in parallel into data/sample/train_series/<study>/<series>/<sop>.dcm.

    python src/download_sample.py --studies 20
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

COMP = "rsna-knee-abnormality-detection"
OUT = Path(__file__).resolve().parents[1] / "data" / "sample"


def list_study_files(api, n_studies):
    """Return {study_uid: [(file_name, bytes), ...]} for the first n_studies train studies."""
    studies, token = {}, None
    while True:
        page = api.competition_list_files(COMP, page_token=token, page_size=1000)
        for f in page.files:
            if not f.name.startswith("train_series/"):
                continue
            study = f.name.split("/")[1]
            if study not in studies and len(studies) == n_studies:
                return studies  # the listing is sorted, so this study's files are all past this point
            studies.setdefault(study, []).append((f.name, int(f.total_bytes or 0)))
        token = page.next_page_token
        if not token:
            return studies


def download(api, name, size, retries=8):
    """Download one slice; skip it if already complete, back off when Kaggle rate-limits (HTTP 429)."""
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            api.competition_download_file(COMP, name, path=str(dest.parent), force=True, quiet=True)
            return
        except HTTPError as e:
            if e.response is None or e.response.status_code != 429 or attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)  # more than this triggers Kaggle's rate limit
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()
    studies = list_study_files(api, args.studies)
    files = [f for fs in studies.values() for f in fs]
    total = sum(b for _, b in files)
    print(f"{len(studies)} studies, {len(files)} slices, {total / 1e9:.2f} GB")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(studies, indent=1))
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda f: download(api, *f), files), 1):
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)} downloaded", flush=True)


if __name__ == "__main__":
    main()
