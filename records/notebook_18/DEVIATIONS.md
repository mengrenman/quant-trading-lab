# Notebook 18: deviations after the freeze

*A working record, written as each change was made on 2026-11-15, in the order made: first the code cells changed in
labels and drawing (17 by the end), then, last, the correction of `lob/ladder.py`, which changes what the library
computes. The paths it names (`cells/`, `after/`, `ladder_fix/`, `smoke/`, `next_scratch/`) are those of the build
folders outside the repository; HANDOFF.md, section 3W.14, says where they are kept. The notebook's own limitations
cell is the finished account.*

2026-11-15. Specification: `LEAD_DECISIONS_AFTER_RUN.md`, "Listed deviations: drawing and labels only", items 1 to 7.
Frozen copies: `.../next_scratch/nb18/frozen_cells/` (96 files). Changed: 16 cell files in `cells/`. Not changed:
`02_predictions.md`, `01_setup.py`, every markdown cell, every judge, collector, pass, bootstrap, fit and constant,
`lob/itch.py`, `lob/feeds.py`, `lob/ladder.py`, the tests. No cell added or removed. No `89_t*` file was touched by the
drawing agent (section 7's own files are not deviations: they are new). **The fix pass (after the audit and the judgment
review) changed one of the 16 again, `81_second_machinery.py`, for the judge's finding 1 (see its row), and put back no
line: the audit found no OTHER line in a frozen cell.** The 19 lines it listed as borderline (ring sizes, the note's
condition, the axis floor, the verdict lookup) each compute a drawn quantity from a result already in memory or choose
where to annotate, are what the lead's items 4 and 7 ask for, and feed no printed number, verdict or gate; they stay.

## How "numbers printed: unchanged" was checked (the same for every row)

- Two smoke runs, the same 96 cells, NVDA and CMCSA, a half hour: `smoke/dev_frozen.executed.ipynb` (the frozen cells) and
  `smoke/dev_changed.executed.ipynb` (this state, from a private copy of the 96 names, so that no `89_t*` file entered). Both: 51 code
  cells run, 0 errors; the changed run ends with a clean `check_namespace()` ("291 names of the setup cell are as it defined
  them"). Tags `dev_frozen`, `dev_changed`.
- `after/dev/compare.py` (report `after/dev/compare_report.txt`): every code cell's text output, line by line, 3,083 lines in
  the frozen run. **Result: 2,947 lines are identical and 136 lines of the frozen run differ**, all of them in one of three ways:
  84 lines hold the same tokens and differ only in column width (a longer column heading re-pads a `to_string` table: cells 22, 25,
  64, 86); 22 lines hold a time or a memory figure (cells 07, 08 rates, 11, 62, 73, 82 and 90: the seconds and memory tables, "In all:",
  and the two run-summary lines of 73 and 82, whose stock-day counts are equal and whose label is the changed one); and 30 are
  the label lines of the rows below (replaced by 48 lines of new wording). The label lines that differ are exactly: the caption
  sentence and the column heading of the imbalance table (cells 22 and 86), of the strips summary (3 settings, cells 25 and 86) and of
  the ladder's lift table (cell 64); the margin line under each VPIN verdict (cells 83 and 90); the two run-summary lines
  (cells 73 and 82). Nothing else differs.
- The data handed to every plotting call (`after/dev/analyze_plotlog.py`, `plotlog_report.txt`): a harness cell (`03_plotlog.py`, in
  neither notebook) wraps `Axes.plot`, `errorbar`, `scatter`, `bar`, `fill_between`, `axhline`, `axvline`, `axhspan` and `text` and
  logs, for each figure, the call and a hash of every array handed to it. 401 calls in the frozen run, 456 in the changed one,
  29 figures each. **Every frozen call is in the changed run with the same arrays, except 3 `text` calls** ("not defined for the
  model" in 77, which moves to the panel's x label by design). The 58 added calls are the rings (36 `scatter`), the notes (10 `text`),
  and the open marks, floor marks and floor line of the headline figure (12; listed per row). Figures of cells I did not change are byte-identical PNGs (8 of 8 checked: 14, 42, 55,
  68, 84_1, 86_0, 87_1, 87_4).
- Lint: the built scratch notebook (`build.py --out after/dev/nb_dev.ipynb --cells-dir after/dev/cells`) passes `ruff check`
  (E, F, W, B, 120 columns) and `ruff format --check` (quotes preserved). Every line of every changed cell is within 120 columns.
- Cases the smoke cannot show are drawn on made-up inputs: `after/dev/draw_cases.py` (output `after/dev/draw_out_changed/`), below.

## Lines changed that are not rows (listed for completeness; the audit found them harmless)

Comment-only edits (`25_imb_strips` 22-24 and 62-64, `33_ofi_figure` 5, `64_a_lift` 10-11, `81_second_machinery` 97-98 and
565-567, and the fix pass's comment on `two_figure`), and the `tight_layout(rect=...)` changes that come with each moved
legend (`28_race`, `33_ofi_figure`, `39_vpin_strips`, `51_trap`, `25_imb_strips`, `73_cal_parameters`, `77_cal_facts`,
`81_second_machinery`). Every `fig.legend` also moves a legend, which the rows above name.

## Rows

### 21_imb_functions.py (item 1; label)
- Before: `'se, strip',` (IMB_COLUMNS), `'se, strip': '.1f',` (IMB_FORMATS), `'se, strip': 100 * strip['lift strip se'] if strip else math.nan,`
- After: the same three lines with `'se, hit less day baseline'`; the comment of `imb_class_table` says what the column is.
- Why that wording and not "less the day's reversal rate": the strip's lift is `hit - max(0.5, the day's reversal rate)` (the
  setup's `imbalance_strip`), so in a class where 50% binds the column is the sd of hit rates; "day baseline" is exact and the
  caption names both baselines. (A small departure from the lead's sentence, for accuracy.)
- Numbers printed: unchanged. The value expression is untouched; compare: the table's lines differ only in column width and heading.

### 22_imb_table.py (item 1; label)
- Before: `... the lift over the reversal rate is bounded by it. se: the lift's block-bootstrap standard error (1,000 draws, ASSUMING EXCHANGEABLE BLOCKS) and the strip's own.`
- After: `se, blocks: the lift's block-bootstrap standard error (...). se, hit less day baseline: NOT the lift's standard error; it is the
  standard error across the half hours (their sd over the root of how many) of each half hour's hit rate less the day's baseline,
  the better of 50% and the day's reversal rate (the one "binds" names). It does not take each half hour's own reversal rate;
  the paired lift's standard error, which does, is in section 7.`
- Numbers printed: unchanged (compare: only the caption's last lines and the heading line differ; 6 table lines re-padded).

### 25_imb_strips.py (item 1 label; item 7 drawing; item 5 legend)
- Label. Before: `'se, strip': ...` (summary column and format key) and `se, strip: the strip's sd over the root of the half hours with readings.`
  After: `'se, hit less day baseline'` and the caption `se, hit less day baseline: NOT the lift's standard error but the sd over the root of the
  half hours with readings of each half hour's hit rate less the day's baseline (the better of 50% and the day's reversal rate),
  which does not take the half hour's own reversal rate; the paired lift's standard error is in section 7.` Numbers unchanged
  (compare: 3 captions, 3 headings, 12 re-padded lines).
- Drawing, `imb_figure` (figure 1 of cell 86, and cell 25's own): before, every `plot_strips` panel kept its own legend (the first
  panel's covered a point) and a half-hour rate made of a handful of readings looked like any other. After: one legend under the
  panels (`fig.legend`, `tight_layout(rect=...)`), and a ring round each point whose area grows with the square root of the half
  hour's readings (`counts` from `imb_strip(r, setting, name, 'n')`, one scale for the whole figure; the legend names the ring and the
  largest count). The ring is a `scatter` at the line's own points: no point is moved or removed (plotlog: 0 frozen calls missing,
  18 `scatter` added a day). The legend move is not on the lead's list for this figure (item 7 names only the readings); it is the
  same defect as item 5 and in the same function: **say so if it is to be reverted.**

### 64_a_lift.py (item 1; label)
- Before: `'se pp (strip)'` (twice), caption `... and the strip's own (sd of the half hours over the root of how many).`
- After: `'se pp (hit less day baseline)'`; caption `"se pp (blocks)" is the lift's block bootstrap ... ; "se pp (hit less day baseline)" is NOT the lift's
  standard error but the sd over the root of how many half hours of each half hour's hit rate less the day's baseline (...; the
  paired lift's standard error is in section 7).`
- Numbers printed: unchanged (compare: 2 captions, 2 headings; 48 re-padded lines of the two tables).

### 81_second_machinery.py (item 2 label; item 4 drawing)
- Item 2. Before: `two_margin(label, value, line, se, strip, how, scale=1.0, digits=1)` ending `; half hours {min} to {max}`; after: a keyword
  `unit='half hours'` (default: every other margin line is as it was) and the VPIN call passes `unit=f'{len(origins)} bucket origins'`,
  so the line ends `; 11 bucket origins 0.197 to 0.219`. Numbers unchanged (compare: 4 lines in 83 and 4 in 90, only the last words).
  The VPIN verdict's sentence is untouched (it comes from the judge).
- Item 4. `two_figure(first, second, day_names=TWO_DAYS)` becomes `two_figure(..., verdicts=None)`; new `TWO_PANEL_TESTS` (which prediction judges which
  panel; asserted equal to `TWO_PANELS`). The first `ax.plot(x + offset, values, ...)` of every panel is exactly the call that was. Added: (a) a
  stock-day whose verdict status (from `two_verdicts_*`, the dicts cell 83 holds) is the judges' own `'not judged'` gets the same mark
  again over the filled one, white inside (open): no gate is computed; (b) on the log panels a value of exactly 0 is drawn as a
  triangle at the panel's floor (`min(positive values and the rule's line) / 4`), a dotted floor line, the note `0: drawn at the floor`
  on the panel, and the lower limit set to a quarter of the floor (an axis's lower limit: the only limit that differs); (c) one legend
  for the figure under the panels: first day, second day, and, only where used, `open mark: the rule does not judge this stock-day
  (its verdict is "not judged")` and `0: drawn at the floor`. The printed headline table is not touched (the zero is in it).
  Numbers printed: unchanged; plotlog: 0 frozen calls missing in this figure, 10 `plot` + 1 `axhline` + 1 `text` added in the smoke.
- Item 4, added by the fix pass (the judgment review's finding 1: the mechanism panel said something false). The panel draws the
  model's mid changes a second as a multiple of the real day's against the line of 0.1, but where one tick holds no more than half the
  time the rule reads the model's *mean spread* instead (AAPL on the first day, AMZN on both): a filled mark there read, beside the new
  open marks, as "judged on this figure", and AMZN's second day, which passed, looked like a failure 20 times over the line. Now a stock-day
  of the mechanism panel whose verdict's own sentence starts "the fitted model's mean spread" (`verdict.detail.startswith(...)`, in the
  panel of the `calibration's mechanism` test only; the verdict is cell 83's, no gate is computed) has its filled mark blanked (a white
  mark over it) and an `x` in the day's color drawn in its place, at the floor if the ratio is 0; the legend gets "x: the rule read this
  stock-day's mean spread (table), not this figure" where one is drawn. The floor note in that panel reads "0 (the model's mid did not move
  in any seed): drawn at the floor" (the other log panels keep "0: drawn at the floor"). The suptitle now says "the dashed line the rule's
  clause in that panel; a stock-day can fail a clause that is not drawn (verdict table)" (the ladder's three failures of the R2 clause sit
  above the drawn quote line), on two lines. The legend takes two rows when it holds more than three entries (`tight_layout(rect=...)`
  moves with it). The `ax.plot(x + offset, values, ...)` of every panel is still the call that was; two `plot` calls are added for each
  x (the blank mark and the x). Drawn on made-up verdicts (`after/fix/draw/draw_two.py`, `out/case_1.png`): the x at the floor, the x off the
  floor, the two-row legend and the two-line note.

### 84_second_figures.py (item 4; drawing)
- Before: `two_figure(two_judged_first, two_judged_second)`. After: `two_figure(two_judged_first, two_judged_second, verdicts=(two_verdicts_first, two_verdicts_second))`.

### 73_cal_parameters.py (items 3, 5, 7; label and drawing)
- Item 3. Before: `ran('Section 5, the first day: ')`; after: `ran('The run so far, at section 5 (the first day): ')`. (The line is the run's running total, 22 stock-days.)
- Items 5, 7. `cal_plot_parameters(results)` becomes `(results, day=DAY)`: the stocks' legend moves from the first panel (over AAPL's first
  point) to the empty eighth panel (`axes[1, 3].legend(..., ncols=3, loc='upper left')`, above the key's text); the suptitle gains `, {day_text(day)}`.
- Numbers printed: unchanged (compare: the one run-summary line, whose stock-day count is equal and whose memory and core-minutes are timing).

### 82_second_run.py (item 3; label)
- Before: `ran('Section 6: ')`; after: `ran('The run so far, at section 6: ')`. Numbers unchanged (14 stock-days in the smoke, both runs).

### 28_race.py (items 5, 7; drawing and label)
- Before: each panel's `ax.legend(fontsize=6, loc='lower right')` (over the binned points) with the label `a = 0.35, c = 1`.
- After: no panel legend; the curve's label is `fitted amplitude a at c = 1 (its value in each panel)`, one `fig.legend` under the panels, and
  `a = {a:.2f}, c = 1` as a text in the panel's upper left; where no bin has `IMB_BIN_MIN` readings the panel says `no bin has 100
  readings` (the entry for binned points with no points, in cell 86's second-day figure). Plotlog: no call missing; 5 `text` calls added a day.
  Numbers printed: unchanged (the cell prints nothing of the figure).

### 33_ofi_figure.py (item 5; drawing)
- Before: the first panel's `ncols=5` legend sat on the y label and the data. After: legends removed from every panel, one `fig.legend` under the panels.

### 39_vpin_strips.py (item 5; drawing)
- Before: a legend in each panel (covering the top of AMZN's line in the first, INTC's lowest point in the second). After: removed; one `fig.legend` under the panels.

### 51_trap.py (item 5; drawing)
- Before: `axes[0].legend(fontsize=6, loc='upper left')` (over INTC's next-unit point at about 80%). After: one `fig.legend` under the panels (four entries).

### 65_a_ofi.py (item 6; label)
- Before: y label `median half-hour R2 at {OFI_MAIN} s`; after: `R2 of each half hour, {OFI_MAIN} s` (each point is one half hour's R2; the medians are in the table above).

### 46_resilience_strips.py (item 7; drawing, annotation)
- After the suptitle's last line: `No strip, not drawn: INTC, CSCO (no half hour has 30 episodes)`, from `results` and `shown_strips` (the stocks the function already
  skipped); nothing is added when every stock has a strip.
  A stock with a strip of a single half hour (INTC on the second day) is not annotated: see "Not made".

### 75_cal_bound_variants.py (item 7; drawing)
- `cal_plot_profile(results)` becomes `(results, day=DAY)`; the title gains `, {day_text(day)}`; the x label breaks after `B (imbalance_offset)` so that it is not cut at the figure's edge.
  The y ticks are not changed (see "Not made").

### 77_cal_facts.py (item 7; drawing; two additions)
- `cal_plot_facts(stock, detail)` becomes `(stock, detail, day=DAY)`; the suptitle gains `, {day_text(day)}`; `not defined for the model` was an `ax.text` at the
  panel's centre (over the real line in three panels) and is now the panel's x label in the model's orange (frozen `text` calls missing: 3, by design).
- Additions beyond the list (flag): the legend of the facts figure moves from the first panel (over the real line) to one `fig.legend` under the panels;
  `cal_plot_depth` and `cal_plot_price` (which had no title) get a suptitle with the date (and the exploration label where it applies), so that every
  figure of the cell carries it.

## Not made

1. **The `-10^-0` y ticks of the profile figure (75).** They are not there: the first run's PNG, enlarged, reads `−10^−1`, `−10^0`, `−10^1`, and a probe
   of a symlog axis with the same settings gives `-10^{0}`. The numbers audit's reading was of a small PNG. Nothing to change.
2. **The mechanism panel's AAPL first-day point (mid changes a second x0.35, line 0.1)** was not made by the drawing agent (it saw only AAPL
   and thought a gate was needed). It is made now, by the fix pass, from the verdict's own sentence (see the row of 81, item 4, added):
   AAPL's first day and AMZN's two days are drawn as an x. Nothing is left here.
3. **INTC's second-day resilience strip of one half hour** (a single point on an axis of 0.20 to 0.22): not on the list, which names only stocks with no strip.
4. **`ran('Section 1: ')` in cell 11.** It is section 1's own total (the first caller of `run_stocks`), so the label is right.
5. **`strip se` of the order-to-trade table (42) and `strip se` of VPIN (39)**: each is a different statistic (the sd of that figure's own half-hour
   values over their root) and is described correctly where it is printed.
6. **Markdown that still calls the strip's standard error the lift's** (prose, written later; listed so that it is not missed): `26_imb_prose2.md` (the TODO of
   the strips: "the lift's two standard errors (... and the strip's own)"), `64_b_lift_prose.md` line 1 ("the strip's own standard error beside the bootstrap's"),
   `60_a_intro.md` line 32 ("beside the strip's own").
7. **Rings' reading for a handful**: where a half hour's value is made of a few readings the ring is as small as the dot, which is the information
   (area follows the readings); no threshold ("fewer than k readings") was added to mark it.

## Added by the lead after the fix pass (2026-11-15, 12:05)

### 90_conclusions.py (the section's number; a comment)
- Before: `# Section 7, what the notebook left, in one place. ...`
- After: `# Section 8, what the notebook left, in one place. ...`
- Why: the new section between the second day and the conclusions is Section 7 ("After the run"), which the changed
  captions of cells 22, 25 and 64 already call it; the conclusions are Section 8 (`90_a_conclusions_intro.md` and
  `00_title.md` say so). The line is a comment: the cell prints nothing different.
- Numbers printed: unchanged.

That makes 17 changed cell files of the 50 frozen code cells: the 16 above and this comment.

## Added by the lead before the last run (2026-11-15, 14:35), after the prose's final judge

### 81_second_machinery.py (item 4 again; drawing)
- Before (second run): the note "0 (the model's mid did not move in any seed): drawn at the floor" was an `ax.text` inside
  the panel at its lower right, where it sat on two of the floor's marks and on the floor's dotted line.
- After: the same words are the panel's x label (`ax.set_xlabel(note, fontsize=7)`), under the axis, where no mark is.
- Numbers printed: unchanged (the cell prints none of its own; cell 84's table is untouched).

### 25_imb_strips.py (item 7 again; label)
- Before (second run): the legend said "ring: area grows with the readings in the half hour (the largest is N)".
- After: "ring: area grows with the square root of the half hour's readings (the most: N)", which is what the code
  draws (`s=6 + 80 * np.sqrt(n / largest)`); the drawing is unchanged.
- Numbers printed: unchanged.

Section 7's cells (not frozen; listed for the record): `89_t3_a`, `89_t5_a`, `89_t6_a` and `89_t9_a` now print their
table's letter ("Table 7b.", "Table 7d.", "Table 7e.", "Table 7h.") at the head of the caption, as 7a, 7c, 7f and 7g
already did; `89_t5_a`'s figure is one row of panels with a short y label (a long one set the panels' height).

## The correction of lob/ladder.py (2026-11-15, 17:15; the lead's decision after the sweep's triage)

This one is not a label or a drawing: it changes what the library computes, and so some printed figures.

- What was wrong (found by a mechanical sweep of the library's tests and a reading of the code, on NVDA and CMCSA
  only; `ladder_breaks/triage.md`, `ladder_breaks/review.md`): `lob.itch` writes an execution with price (a printable
  `C`) and a displayed order executed in a cross at the execution's price, while the book falls at the order's level.
  As frozen, `lob.ladder` read such a row's price as its level: (A) `_batched` keyed a window's net change by it, so an
  add and its priced execution in one window did not net to nothing (an update that should not be there) and a change
  could be netted against another level's; (B) `levels` matched a priced fill to the changed level nearest the fill's
  price, which could leave the level it took as a cancel beside the trade.
- The correction (`ladder_fix/ladder.diff`, 117 lines of unified diff; `ladder_fix/STATE.md`): a `Day.level` array,
  read from the book (the level of the row's side whose size fell by the row's size); `_batched` keys by it; `levels`
  matches a priced fill by it. Frozen SHA-256 `d7d44476...a5008`, corrected `396e314c...acf22`. `lob/itch.py` and
  `lob/feeds.py` are as frozen.
- Proved on NVDA and CMCSA, 2019-03-27 (by the fixer and again by the judgment review with code of its own): every
  item of every rung that differs between the frozen file and the corrected one holds a priced or cross row; an
  independent count keyed by the book's level equals the corrected file on every window, and keyed by the row's own
  price equals the frozen file.
- Tests: `tests/test_ladder.py` had 34 tests at the freeze and has 58 now (the sweep's triage added tests for the gaps
  it found; three recorded the two defects as expected failures and now pass; nine state the correction). The pin of
  CMCSA's 50 ms empty windows is 2,786 (2,785 as first built).
- Numbers printed: CHANGED where a batched rung's count of updates or a `levels` figure of the exploration stocks is
  printed. The list of every printed line that differs between the run before the correction (the third) and the run
  after it (the fourth) is `after/compare_third_fourth.txt`; the limitations cell lists each figure before and after.
- Section 7 gains item i (Table 7i, `89_t9_b_z1_after_levels.py`): the count of rows the correction touches on each
  stock-day.
