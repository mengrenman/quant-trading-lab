"""Check notebook 18's freeze from the repository alone.

    python records/notebook_18/check_freeze.py

It reads FREEZE.txt (the SHA-256 of each of the 50 code cells and of lob/itch.py, lob/feeds.py and lob/ladder.py,
recorded before any reported stock was read), first_run.ipynb (the notebook as the first run left it) and the
notebook and the library as they are now, and prints: whether the first run's code cells are the frozen ones;
whether the combined hash is the recorded one; which code cells of the notebook are still the frozen ones, which
were changed and which are new; and which of the three library files is as frozen. It changes nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def code_cells(path: Path) -> list[str]:
    # a cell file's text is the cell's source and a final newline, which is what the freeze hashed
    cells = json.loads(path.read_text())['cells']
    return [sha((''.join(c['source']) + '\n').encode()) for c in cells if c['cell_type'] == 'code']


def main() -> None:
    lines = (HERE / 'FREEZE.txt').read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('Required:'))
    end = next(i for i, line in enumerate(lines) if line.startswith('Combined SHA-256'))
    required = [line.split() for line in lines[start + 1 : end] if line.strip()]
    frozen_cells = [digest for digest, name in required if '/cells/' in name]
    frozen_library = {name.split('/lob/')[-1]: digest for digest, name in required if '/cells/' not in name}
    combined = sha(''.join(digest for digest, _ in required).encode())
    print(f'combined hash of the {len(required)} recorded hashes: {combined}')
    print(f'  as FREEZE.txt records it: {combined in lines[end]}')

    first = code_cells(HERE / 'first_run.ipynb')
    print(f'first_run.ipynb: {len(first)} code cells; the frozen ones, in order: {first == frozen_cells}')

    now = code_cells(REPO / 'notebooks' / '18_real_books.ipynb')
    same = sum(digest in frozen_cells for digest in now)
    print(
        f'notebooks/18_real_books.ipynb: {len(now)} code cells; {same} are frozen cells byte for byte, '
        f'{len(frozen_cells) - same} frozen cells were changed (DEVIATIONS.md lists each) and '
        f'{len(now) - len(frozen_cells)} are new (section 7)'
    )
    for name, digest in frozen_library.items():
        current = sha((REPO / 'lob' / name).read_bytes())
        changed = f'changed (now {current[:8]}; ladder_correction.diff is the change)'
        state = 'as frozen' if current == digest else changed
        print(f'lob/{name}: frozen {digest[:8]}, {state}')


if __name__ == '__main__':
    main()
