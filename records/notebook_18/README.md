# Notebook 18: the records of its freeze and of what changed after it

Notebook 18 ("Real Books") fixed its code before any reported stock was read, ran it once, and promised that any change
after that would be listed with its figures before and after. These files are the "before", kept in the repository so
that the notebook's account can be checked from the repository alone:

```
python records/notebook_18/check_freeze.py
```

| File | What it is |
|---|---|
| `FREEZE.txt` | The freeze, recorded 2026-11-15 at 09:32:55 +08:00, before any reported stock or anything of the second day had been converted or read: the SHA-256 of each of the 50 code cells and of `lob/itch.py`, `lob/feeds.py` and `lob/ladder.py`, and the hash of those 53 hashes together (`2b220ac5...`). The paths are those of the build folder; a cell file's text is the notebook's code cell and a final newline. |
| `first_run.ipynb` | The notebook as the first run left it (09:33:24 to 10:21:46 that day): the frozen code and what it printed, with the prose still drafts. Every verdict of notebook 18 was first read from it. It is a record: it is never rebuilt or run again, and no make target touches it. |
| `DEVIATIONS.md` | The working list of every change after the freeze, in the order made: the 17 code cells changed in labels and drawing, with how "no number changed" was checked, and, last, the correction of `lob/ladder.py`. |
| `ladder_correction.diff` | That correction: the frozen `lob/ladder.py` against the corrected one. |
| `compare_first_fourth.txt` | The first run's outputs against the fourth run's, which the notebook holds, cell by cell and line by line. Of the 4,544 lines the 50 frozen cells printed in the first run, 285 differ: 168 in column width only, 60 label lines, 43 times and memory figures, 2 a joblib warning, and 12 that the correction of `lob/ladder.py` moved. |
| `check_freeze.py` | Reads the files above and the notebook and library as they are now, and prints whether the first run's code cells are the frozen ones, which of the notebook's code cells still are (33), which were changed (17) and which are new (10), and which of the three library files is as frozen (`lob/itch.py` and `lob/feeds.py`; `lob/ladder.py` was corrected). |

The notebook's limitations cell tells all of this in full, with every figure the correction moved. The rest of the
records (each run's notebook and outputs, the reviews, the cell files, the correction's proof on the two exploration
stocks) are outside the repository; HANDOFF.md, section 3W.14, says where.

The figures in these files are derived from Nasdaq's sample files, which carry no stated terms of use: like the
notebook itself, they stay in this private repository and are not published.
