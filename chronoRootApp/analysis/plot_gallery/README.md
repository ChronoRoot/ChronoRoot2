# Plot gallery harness

Builds a **commentable figure folder** from a local ChronoRoot report `data/`
directory, remixed into invented plate conditions and run labels. Generates
**figure kinds**, not a full project report.

No machine-specific paths are baked in: you must pass `--source`.

## Quick start

```bash
cd chronoRootApp
conda activate ChronoRoot2
python -m analysis.plot_gallery.generate --list
python -m analysis.plot_gallery.generate \
  --source /path/to/YourProject/Report/data
# one scenario / kind:
python -m analysis.plot_gallery.generate \
  --source /path/to/YourProject/Report/data \
  --scenario g4_p2_e2 --kinds temporal
# relative-measurement titles (writes under <scenario>/relative/):
python -m analysis.plot_gallery.generate \
  --source /path/to/YourProject/Report/data \
  --scenario g4_p2_e2 --kinds temporal,overview --relative --keep
```

`--source` must contain at least:

- `Temporal_Data.csv`
- `Germination_Time.csv`
- `Convex_Hull_Data.csv`

Output defaults to `ChronoRoot2/plot_gallery/` (gitignored). See that folder’s `INDEX.md`.

## What is remixed

Donor trajectories come from your `--source` tables. All factor labels are invented
(original genotype names are never kept):

- Genotypes: `Genotype A` … `Genotype F`
- Plates: `MS`, `MS+Cd`, `MS+NaCl`, `MS+PEG` (up to 4)
- Extra / Run: `Run A`, `Run B`

## Figure kinds

| kind | metric used | notes |
|------|-------------|-------|
| `temporal` | MainRootLength | every applicable comparison mode + count companions |
| `scalar` | Germination time | box + swarm per mode |
| `interval` | Convex Hull Area | box + swarm at a few Day snapshots |
| `overview` | multi-metric | genotype overview grid |

Use the `.svg` beside each `.png` to judge Day / title spacing.
