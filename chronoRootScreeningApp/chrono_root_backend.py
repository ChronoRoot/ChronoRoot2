"""Import chronoRootApp analysis modules without modifying that package.

Screening and the standard app share this backend: put chronoRootApp on
sys.path once, then import analysis.* and gui.stats_config_dialog as usual.
"""

import os
import sys

SCREENING_APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHRONOROOT_APP_DIR = os.path.abspath(
    os.path.join(SCREENING_APP_DIR, '..', 'chronoRootApp')
)
# Screening must stay first so plant_viewer / qr / calibration_helper
# resolve here, not to the same names under chronoRootApp.
for _path in (SCREENING_APP_DIR, CHRONOROOT_APP_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, SCREENING_APP_DIR)
sys.path.insert(1, CHRONOROOT_APP_DIR)

from analysis.dataWork import dataWork  # noqa: E402
from analysis.report import (  # noqa: E402
    generateTableTemporal,
    performStatisticalAnalysis,
    performStatisticalAnalysisForMetrics,
    plot_individual_plant,
    plot_info_all,
)
from analysis.stats_utils import (  # noqa: E402
    ensure_factor_columns,
    get_enabled_comparison_modes,
    log_auto_disabled_modes,
    perform_scalar_pairwise_stats,
)
from analysis.report_plots import emit_scalar_comparison_plots  # noqa: E402
from analysis.fpca_analysis import performFPCA  # noqa: E402
from analysis.fourier_analysis import makeFourierPlots  # noqa: E402
from analysis.time_windows import (  # noqa: E402
    apply_time_windows,
    hourly_covers_analysis_period,
    hourly_files_exist,
    load_hourly_tables,
    subtract_initial_stage,
)
from analysis.utils.fileUtilities import (  # noqa: E402
    convertToPathSafe,
    normalize_factor_value,
    UNSPECIFIED_FACTOR,
)
from analysis.utils.report_paths import (  # noqa: E402
    MODULE_TEMPORAL,
    data_file,
    metric_dir,
    metric_slug,
    purge_disabled_comparison_outputs,
    report_root,
    temporal_metric_slug,
)
