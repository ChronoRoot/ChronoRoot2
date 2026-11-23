# ChronoRoot 2.1 - Standard Root Phenotyping Interface

This directory contains the Standard Root Phenotyping Interface for ChronoRoot 2.1, designed for detailed architectural analysis of individual plant root systems.

![Figure](Screenshots/MainInterface.png)

## Overview

The Standard Interface maintains continuity with earlier ChronoRoot releases while adding modern visualization, multi-factor experimental labels, coordinated analysis periods, plate-scale quality assurance, and a reworked comparison-mode report.

## Key Features

- Detailed RSA analysis through graph-based representation
- Temporal tracking of root development (improved lateral-root tracking)
- Plate condition and extra variable (run / replica) metadata per plant
- Analysis-period / time-group coordination across imaging cohorts
- Per-video plate scale registry with Test QR and manual Set scale
- Real-time visualization of segmentation results
- Reworked statistical analysis with advanced comparison modes
- Automated report generation (selectable subsets, paired plot/stats, FPCA/Fourier showcases)
- Export in Root System Markup Language (RSML) format

## What's new in 2.1 (Standard App)

### Experimental factors
- **Genotype** (plant identifier), **plate condition**, and **extra variable** are stored in plant metadata and used throughout postprocess and report.
- Axis-label fields on the Generate Report tab rename Genotype / plate / extra on figures and stats headers. Colors are stable across modules.

### Statistical analysis & reports
- **Configure Report Parameters** enables advanced comparison modes (default on): genotypes across all data, genotypes within plate or extra, plate/extra directly, and plate/extra within genotype.
- Each mode writes paired `{metric}_{mode}.png` and `{metric}_{mode}_stats.txt` files; disabled modes are purged on regenerate.
- Overviews (temporal `all_metrics_*`, convex atlases, FPCA and Fourier showcases) follow the same modes: pooled when comparing all genotypes, plus `by_plate/` / `by_extra/` when those genotype-within-condition modes are enabled. Fits (e.g. FPCA) still use all plants; showcases filter for display.
- Optional SVG/PDF export, relative measurements to the initial stage, germination-time analysis, and report subset selection (who is included in a named Report folder).
- Lateral-angle stats and overlays skip plants with no measured laterals.

### Time coordination
- **Analysis period** dialog defines time groups (clock or elapsed), duration, preview alignment, and whether partially overlapping cohorts are merged or kept as separate 12 h start cohorts.
- Figures use consistent elapsed-time and Day axes with optional clock tick marks.

### Plate scale / QR QA
- **Review plate calibrations** lists per-video scale, QR vs manual sources, and warnings.
- **Test QR** and **Set scale** write into `calibrations.json`; postprocess prefers the registry.
- Image discovery matches Preview (including `Segmentation/segmentation_metadata.json` → `input_path` when PNGs are stored outside the project tree).

## Directory Structure

```
chronoRootApp/
├── analysis/                  # Core analysis modules
│   ├── graphUtils/            # Graph representation utilities
│   ├── imageUtils/            # Image processing utilities
│   ├── rsmlUtils/             # RSML export functionality
│   └── utils/                 # Utilities (report_paths, calibration_registry, …)
├── gui/                       # Dialogs (stats config, time windows, plate scale, …)
├── placeholder_figures/       # Default images for the interface
├── Screenshots/               # Interface screenshots for documentation
├── 1_analysis.py              # Plant selection and analysis workflow
├── 2_postprocess.py           # Post-processing of analysis results
├── 3_generateReport.py        # Report generation functionality
├── plant_viewer.py            # Quality control and review tools
├── 5_rerun_all_analysis.py    # Batch reprocessing utility
├── calibration_helper.py      # Calibration tool implementation
├── default.json               # Default parameters
└── run.py                     # Main interface implementation
```

### Analysis output (per plant)

Per-plant tracking results are stored under:

```
{Project}/Analysis/{Experiment}/{rpi}/cam_{N}/plant_{N}/Results_{j}/
```

The GUI shows one row per plant slot and always uses the latest `Results_*` folder. Older runs are removed automatically after a successful repeat/redo analysis.

### Report output (metric-first)

Generated reports are organized by biological metric under `{Project}/Report/` (or a named subset folder). Each comparison mode produces a paired plot and stats file (`{metric_slug}_{mode}.png` + `{metric_slug}_{mode}_stats.txt`).

```
Report/
├── data/                              # Aggregated CSV inputs
├── individual_plots/{experiment}/     # Per-plant growth curves
├── temporal_parameters/
│   ├── overview/                      # all_metrics_*; optional by_plate/ / by_extra/
│   ├── main_root_length/
│   │   ├── main_root_length_by_genotype.png
│   │   ├── main_root_length_by_genotype_stats.txt
│   │   ├── summary_table.csv
│   │   ├── fpca/                      # overview + by_plate/by_extra showcases; pc{n}/ stats
│   │   └── growth_speed/              # Fourier joint/individual + mode plots
│   └── ...
├── convex_hull/
│   ├── overview/                      # atlases; optional by_plate/ / by_extra/
│   └── {metric_slug}/
└── angles/
    ├── mean_emergence_angle/
    ├── first_lr_tip_angle/
    └── overlays_{experiment}/
```

## Measurements Provided

### Basic Architecture
- Main Root (MR) Length
- Lateral Root (LR) Length
- Total Root (TR) Length
- Number of Lateral Roots
- Discrete LR Density
- Main Over Total Root ratio
- Hypocotyl Length
- Germination time (first leave of main-root length from zero)

### Growth Analysis
- Growth Speed
- Fourier Components
- FPCA of temporal trajectories

### Spatial Distribution
- Convex Hull Area
- Convex Hull Width
- Convex Hull Height
- Root Density
- Aspect Ratio

### Angular Measurements
- Tip Angle
- Emergence Angle
- First lateral tip-angle decay

## Usage

### Starting the Interface

```bash
conda activate ChronoRootInterface
python run.py
```

Suggested sequence for a new experiment: set project and video → assign genotype / plate / extra → review plate calibrations (Test QR / Set scale) → define analysis periods → analyze plants → postprocess → Configure Report Parameters → Review and Report.

## Integration with Segmentation

This interface works with segmentation masks produced by the nnUNet module in `segmentationApp/`. For best results, ensure that videos are properly segmented before analysis.
