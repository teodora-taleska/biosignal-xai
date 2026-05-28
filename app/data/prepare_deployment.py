"""
Copy curated signal files and FCN-Wang checkpoint into the app package
so the app works both locally and on Streamlit Community Cloud without
the full PTB-XL dataset.

Run once from the repo root (requires the full data/ directory):
    python app/data/prepare_deployment.py

What it does:
    1. Reads app/data/curated_200.json for the 200 curated record paths.
    2. Copies {DATA_DIR}/{filename_lr}.hea and .dat
          -> app/data/signals/{filename_lr}.hea / .dat
       (preserving the records100/xxxxx/xxxxx_lr directory structure).
    3. Copies results/fcn_wang_baseline/checkpoint.pt
          -> app/model/checkpoint.pt

After running, commit the two new directories:
    git add app/data/signals/ app/model/checkpoint.pt
    git commit -m "add embedded signals and checkpoint for cloud deployment"
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_FILE    = Path(__file__).resolve()
APP_DATA = _FILE.parent          # app/data/
APP_DIR  = APP_DATA.parent       # app/


def _find_dir(start: Path, name: str) -> Path:
    candidate = start / name
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find '{name}/' above {start}")


def _find_file(start: Path, rel: str) -> Path:
    candidate = start / rel
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find '{rel}' above {start}")


REPO_ROOT   = _FILE.parents[2]
DATA_DIR    = _find_dir(REPO_ROOT, 'data')
CKPT_SRC    = _find_file(REPO_ROOT, 'results/fcn_wang_baseline/checkpoint.pt')
SIGNALS_DST = APP_DATA / 'signals'
CKPT_DST    = APP_DIR  / 'model' / 'checkpoint.pt'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=== prepare_deployment.py ===\n')

    # Load curated index
    with open(APP_DATA / 'curated_200.json') as f:
        curated: list[dict] = json.load(f)

    # 1. Copy WFDB signal files
    print(f'Copying {len(curated)} signal records to {SIGNALS_DST} ...')
    copied = skipped = 0
    for rec in curated:
        fl = rec['filename_lr']   # e.g. 'records100/00000/00001_lr'
        for ext in ('.hea', '.dat'):
            src = Path(str(DATA_DIR / fl) + ext)
            dst = Path(str(SIGNALS_DST / fl) + ext)
            if not src.exists():
                print(f'  MISSING: {src}', file=sys.stderr)
                continue
            if dst.exists():
                skipped += 1
            else:
                _copy(src, dst)
                copied += 1

    print(f'  {copied} files copied, {skipped} already present.\n')

    # 2. Copy FCN-Wang checkpoint
    print(f'Copying checkpoint  {CKPT_SRC}')
    print(f'                 -> {CKPT_DST}')
    if CKPT_DST.exists():
        print('  Already present, skipping.')
    else:
        _copy(CKPT_SRC, CKPT_DST)
        print('  Done.')

    print('\n=== All done ===')
    print('\nNext steps:')
    print('  git add app/data/signals/ app/model/checkpoint.pt')
    print('  git commit -m "add embedded signals and checkpoint for cloud deployment"')


if __name__ == '__main__':
    main()
