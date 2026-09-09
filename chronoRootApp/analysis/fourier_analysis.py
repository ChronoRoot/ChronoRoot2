from .utils import report_utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import signal
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from .utils.fileUtilities import convertFromPathSafe, load_result_metadata, normalize_factor_value
from .stats_utils import (
    perform_fourier_pairwise_stats,
    ensure_factor_columns,
    comparison_modes_for_run,
    iter_replica_runs,
    _mode_spec,
    _describe_averaging,
)
from .report_plots import plot_comparison_mode
from .utils.report_paths import (
    MODULE_TEMPORAL,
    FOURIER_PARENT_METRICS,
    REPLICAS_DIR,
    analysis_dir,
    clear_replicas_subdir,
    comparison_plot_path,
    data_file,
    plot_file,
    stats_file,
)
from .plot_time_axis import decorate_elapsed_time_axis
from .time_windows import elapsed_hour_windows
from .utils.report_style import genotype_palette_for_data, get_genotype_axis_label


def _odd_kernel(n, requested):
    n = int(n)
    if n < 3:
        return 0
    limit = n if n % 2 else n - 1
    k = min(int(requested), limit)
    if k % 2 == 0:
        k -= 1
    return k if k >= 3 else 0

def _interp_interior_nans(values):
    """Fill NaNs between the first and last finite samples; keep end pads as NaN."""
    y = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return y
    first = int(np.argmax(finite))
    last = len(y) - 1 - int(np.argmax(finite[::-1]))
    seg = y[first:last + 1]
    fin = np.isfinite(seg)
    if not fin.all():
        idx = np.arange(len(seg))
        seg = seg.copy()
        seg[~fin] = np.interp(idx[~fin], idx[fin], seg[fin])
        y[first:last + 1] = seg
    return y


def _fft_ready_signal(values):
    """Drop leading/trailing NaN pads; interpolate interior gaps. None if too short."""
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return None
    first = int(np.argmax(finite))
    last = len(y) - 1 - int(np.argmax(finite[::-1]))
    interior = _interp_interior_nans(y)[first:last + 1]
    if not np.isfinite(interior).all() or len(interior) < 2:
        return None
    return interior


def _column_has_signal(df, column):
    if df is None or df.empty or column not in df.columns:
        return False
    series = pd.to_numeric(df[column], errors='coerce')
    finite = series.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return float(np.nanmax(np.abs(finite))) > 1e-9


class MetricConfig:
    """Configuration class for different metrics"""
    METRICS = {
        'MR': {
            'column': 'MainRootLengthGrad (mm/h)',
            'title': 'Main Root Growth Speed',
            'ylabel': 'Speed (mm/h)',
            'norm_ylabel': 'Normalized Speed'
        },
        'TR': {
            'column': 'TotalLengthGrad (mm/h)',
            'title': 'Total Root Growth Speed',
            'ylabel': 'Speed (mm/h)',
            'norm_ylabel': 'Normalized Speed'
        },
        'HY': {
            'column': 'HypocotylLengthGrad (mm/h)',
            'title': 'Hypocotyl Growth Speed',
            'ylabel': 'Speed (mm/h)',
            'norm_ylabel': 'Normalized Speed'
        }
    }

    @classmethod
    def get_config(cls, metric_type: str) -> dict:
        """Get configuration for a specific metric"""
        if metric_type not in cls.METRICS:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        return cls.METRICS[metric_type]

class DataProcessor:
    """Class for processing and analyzing growth data"""
    
    def __init__(self, conf: dict):
        self.conf = conf

    def _prepare_signal(self, values, normalize=False, detrend=False, medfilt=False):
        m_speed = np.array(values, copy=True, dtype=float)
        valid = np.isfinite(m_speed)
        if valid.sum() < 2:
            return m_speed
        filled = _interp_interior_nans(m_speed)
        first = int(np.argmax(valid))
        last = len(m_speed) - 1 - int(np.argmax(valid[::-1]))
        work = filled[first:last + 1].copy()
        if not np.isfinite(work).all():
            return np.where(valid, m_speed, np.nan)
        if normalize:
            mean = np.mean(m_speed[valid])
            std = np.std(m_speed[valid])
            work = (work - mean) / std if std != 0 else work - mean
        if medfilt:
            n = len(work)
            k5 = _odd_kernel(n, 5)
            k25 = _odd_kernel(n, 25)
            if k5:
                work = signal.medfilt(work, k5)
            if k25:
                work = work - signal.medfilt(work, k25)
        if detrend:
            work = signal.detrend(work)
        out = np.full(m_speed.shape, np.nan, dtype=float)
        out[first:last + 1] = work
        return np.where(valid, out, np.nan)

    def read_and_process_data(self, temporal_df: pd.DataFrame, root: str = 'MainRootLengthGrad (mm/h)',
                             normalize: bool = False, detrend: bool = False,
                             medfilt: bool = False) -> pd.DataFrame:
        """Build aligned growth-speed series from Temporal_Data."""
        if temporal_df is None or temporal_df.empty:
            raise ValueError("No temporal data for Fourier analysis")
        if root not in temporal_df.columns:
            raise ValueError(f"Column {root} is not in Temporal_Data")

        work = ensure_factor_columns(temporal_df.copy())
        if 'Plant_id' not in work.columns:
            raise ValueError("Temporal_Data is missing Plant_id")
        key_cols = ['Experiment', 'Plant_id']
        if 'Video' in work.columns:
            key_cols = ['Experiment', 'Video', 'Plant_id']
        work = work.sort_values(key_cols + ['ElapsedTime (h)'])

        valid_datasets = []
        for i, (_key, group) in enumerate(work.groupby(key_cols, sort=False)):
            group = group.sort_values('ElapsedTime (h)')
            time_arr = pd.to_numeric(group['ElapsedTime (h)'], errors='coerce').to_numpy()
            signal_arr = self._prepare_signal(
                group[root].to_numpy(),
                normalize=normalize, detrend=detrend, medfilt=medfilt,
            )
            if not np.isfinite(signal_arr).any():
                continue
            dates = group['Date'].to_numpy() if 'Date' in group.columns else [pd.NaT] * len(signal_arr)
            valid_datasets.append({
                'exp': str(group['Experiment'].iloc[0]) if 'Experiment' in group.columns else str(_key[0] if isinstance(_key, tuple) else _key),
                'signal': signal_arr,
                'time': time_arr,
                'date': dates,
                'meta': {
                    'Experiment': str(group['Experiment'].iloc[0]) if 'Experiment' in group.columns else 'unspecified',
                    'PlateCondition': normalize_factor_value(group['PlateCondition'].iloc[0]) if 'PlateCondition' in group.columns else 'unspecified',
                    'ExtraVariable': normalize_factor_value(group['ExtraVariable'].iloc[0]) if 'ExtraVariable' in group.columns else 'unspecified',
                },
            })

        if not valid_datasets:
            raise ValueError("No valid data processed from Temporal_Data")

        all_data = []
        for item in valid_datasets:
            meta = item.get('meta', {})
            df = pd.DataFrame({
                'Time': item['time'],
                'ElapsedTime (h)': item['time'],
                'Date': item['date'],
                'Signal': item['signal'],
                'Type': item['exp'],
                'i': len(all_data),
                'Experiment': meta.get('Experiment', item['exp']),
                'PlateCondition': meta.get('PlateCondition', 'unspecified'),
                'ExtraVariable': meta.get('ExtraVariable', 'unspecified'),
            })
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        fft_data = []
        for (exp_type, i), group in combined_df.groupby(['Type', 'i']):
            fft_input = _fft_ready_signal(group['Signal'].values)
            if fft_input is None:
                continue
            freqs = np.fft.fftfreq(len(fft_input), d=1)
            fft_vals = np.abs(np.fft.fft(fft_input))
            fft_df = pd.DataFrame({
                'Freqs': freqs,
                'FFT': fft_vals,
                'Type': exp_type,
                'i': i,
                'Experiment': group['Experiment'].iloc[0],
                'PlateCondition': group['PlateCondition'].iloc[0],
                'ExtraVariable': group['ExtraVariable'].iloc[0],
            })
            fft_data.append(fft_df)
        if fft_data:
            combined_df = pd.concat([combined_df, pd.concat(fft_data, ignore_index=True)], ignore_index=True)
        return combined_df
    
    def perform_statistical_analysis(self, data_orig: pd.DataFrame, data_detrended: pd.DataFrame, metric_type: str):
        """Perform statistical analysis on temporal and frequency data."""
        try:
            slug = FOURIER_PARENT_METRICS[metric_type]
            metric_label = MetricConfig.get_config(metric_type)['title']
            growth_dir = analysis_dir(self.conf, MODULE_TEMPORAL, slug, 'growth_speed')
            data_orig = ensure_factor_columns(data_orig)
            data_detrended = ensure_factor_columns(data_detrended)
            dt = int(self.conf['everyXhourFieldFourier'])
            time_data = data_orig[data_orig['Time'].notna()]
            fft_detrended = data_detrended[data_detrended['Freqs'].notna()]
            target_rhythms = {'24h Period': 1 / 24, '12h Period': 1 / 12}

            for mode in comparison_modes_for_run(self.conf):
                self._emit_growth_speed_mode(
                    self.conf, mode, slug, metric_type, metric_label,
                    time_data, fft_detrended, dt, target_rhythms, growth_dir,
                    'growth_speed',
                )

            clear_replicas_subdir(growth_dir)
            fft_extra = fft_detrended['ExtraVariable'].astype(str) if 'ExtraVariable' in fft_detrended.columns else None
            for extra, extra_slug, subset, replica_conf in iter_replica_runs(self.conf, time_data):
                replica_dir = analysis_dir(
                    self.conf, MODULE_TEMPORAL, slug, 'growth_speed', REPLICAS_DIR, extra_slug,
                )
                if fft_extra is None:
                    subset_fft = fft_detrended.iloc[0:0]
                else:
                    subset_fft = fft_detrended[fft_extra == str(extra)]
                for mode in comparison_modes_for_run(replica_conf):
                    self._emit_growth_speed_mode(
                        replica_conf, mode, slug, metric_type, metric_label,
                        subset, subset_fft, dt, target_rhythms, replica_dir,
                        'growth_speed', REPLICAS_DIR, extra_slug,
                    )

        except Exception as e:
            print(f"Error in statistical analysis: {str(e)}")

    def _emit_growth_speed_mode(
        self, conf, mode, slug, metric_type, metric_label,
        time_data, fft_detrended, dt, target_rhythms, growth_dir, *stats_subpath,
    ):
        spec = _mode_spec(mode, conf=conf)
        stats_path = stats_file(conf, MODULE_TEMPORAL, slug, mode, *stats_subpath)
        with open(stats_path, 'w') as f:
            f.write(f'CHRONOROOT 2.0 STATISTICAL REPORT - {metric_type}\n')
            f.write('=' * 60 + '\n')
            f.write('Using Mann Whitney U test to compare groups\n')
            f.write(f"{spec['header']}\n")
            f.write(f'{_describe_averaging(conf)}\n')
            f.write('PART 1: HOURLY GROWTH SPEED COMPARISONS (Original Data)\n')

            for start, end in elapsed_hour_windows(time_data, dt, hour_col='Time'):
                subdata = time_data[time_data['Time'].isin(np.arange(start, end))]
                f.write(f'\nWindow: {start}h to {end}h\n')
                perform_fourier_pairwise_stats(
                    conf, subdata, 'Signal', f,
                    plant_id_col='i', type_col='Type', modes=[mode],
                )

            f.write('\n' + '=' * 60 + '\n')
            f.write('PART 2: CIRCADIAN RHYTHM ANALYSIS (Detrended/Normalized FFT)\n')
            f.write('=' * 60 + '\n')

            if fft_detrended is not None and not fft_detrended.empty:
                available_freqs = fft_detrended['Freqs'].unique()
                if len(available_freqs):
                    for label, target_freq in target_rhythms.items():
                        f.write(f'\nFrequency Bin: {label} ({target_freq:.4f} Hz)\n')
                        closest_freq = available_freqs[np.argmin(np.abs(available_freqs - target_freq))]
                        freq_subdata = fft_detrended[fft_detrended['Freqs'] == closest_freq]
                        perform_fourier_pairwise_stats(
                            conf, freq_subdata, 'FFT', f,
                            plant_id_col='i', type_col='Type', modes=[mode],
                        )

        plot_data = ensure_factor_columns(time_data.copy())
        plot_data['ElapsedTime (h)'] = plot_data['Time']
        plot_comparison_mode(
            conf, plot_data, 'Signal', mode,
            comparison_plot_path(growth_dir, mode, metric_slug=slug),
            x_col='Time', metric_label=metric_label,
            module=MODULE_TEMPORAL, metric_slug_name=slug,
            analysis_type='growth_speed',
        )


    def _write_comparison_stats(self, f, subdata: pd.DataFrame, exp1_name: str, exp2_name: str, col='Signal', is_fft=False):
        """Write comparison statistics between two experiments"""
        exp1 = subdata[subdata['Type'] == exp1_name][col]
        exp2 = subdata[subdata['Type'] == exp2_name][col]

        try:
            if len(exp1) == 0 or len(exp2) == 0:
                return

            U, p = stats.mannwhitneyu(exp1, exp2)
            p_val = round(p, 6)

            f.write(f'Comparison: {exp1_name} vs {exp2_name}\n')
            f.write(f'  - Samples: {len(exp1)} vs {len(exp2)}\n')
            f.write(f'  - Mean: {exp1.mean():.4f} vs {exp2.mean():.4f}\n')
            f.write(f'  - Std Dev: {exp1.std():.4f} vs {exp2.std():.4f}\n')
            # Standardized significance notation for the report
            sig_text = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
            stars = "**" if p < 0.001 else ("*" if p < 0.05 else "ns")
            
            metric_label = "FFT Energy" if is_fft else "Speed"
            f.write(f'  - Result: {metric_label} is {sig_text} (p={p_val}, {stars})\n')

        except Exception as e:
            f.write(f'Error comparing {exp1_name} and {exp2_name}: {str(e)}\n')

class Visualizer:
    """Class for creating visualizations"""

    def __init__(self, conf: dict, metric_type: str):
        self.conf = conf
        self.metric_type = metric_type
        self.slug = FOURIER_PARENT_METRICS[metric_type]
        self._setup_plot_style()

    def _clock_ticks(self, ax, data, *, day_axis=True):
        tick_df = data
        if 'ElapsedTime (h)' not in tick_df.columns and 'Time' in tick_df.columns:
            tick_df = tick_df.rename(columns={'Time': 'ElapsedTime (h)'})
        decorate_elapsed_time_axis(ax, tick_df, self.conf, day_axis=day_axis)

    def _setup_plot_style(self):
        """Set up matplotlib plot style"""
        SMALL_SIZE = 10
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

    def create_joint_plot(self, data: pd.DataFrame, data_detrended: pd.DataFrame, 
                         time: np.ndarray, metric_config: dict, 
                         output_prefix: str):
        """Create joint plot with original and detrended data"""
        fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(2, 2)
        
        axes = {
            'original': fig.add_subplot(gs[0, 0]),
            'detrended': fig.add_subplot(gs[0, 1]),
            'fft_original': fig.add_subplot(gs[1, 0]),
            'fft_detrended': fig.add_subplot(gs[1, 1])
        }

        # Plot original data
        self._plot_time_series(axes['original'], data, time, 
                             metric_config['ylabel'], 
                             f"Original {metric_config['title']}")
        
        # Plot detrended data with reference curves
        self._plot_time_series(axes['detrended'], data_detrended, time,
                             metric_config['norm_ylabel'],
                             f"Normalized & Detrended {metric_config['title']}")
        
        # Add reference curves to detrended plot
        exp1 = 1.75 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = 1.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        axes['detrended'].plot(time, exp1, color='red', label='24h rhythm')
        axes['detrended'].plot(time, exp2, color='black', label='12h rhythm')
        
        # Update legend for detrended plot
        handles, labels = axes['detrended'].get_legend_handles_labels()
        axes['detrended'].legend(handles, labels, loc='best')

        # Plot FFTs
        self._plot_fft(axes['fft_original'], data, "FFT of Original Signal")
        self._plot_fft(axes['fft_detrended'], data_detrended, "FFT of Normalized & Detrended Signal")

        plt.suptitle(f"Joint Plot - {metric_config['title']}", fontsize=16, y=1.02)

        for ext in ['png', 'svg']:
            plt.savefig(
                plot_file(self.conf, MODULE_TEMPORAL, self.slug, f'{self.slug}_overview_joint.{ext}', 'growth_speed'),
                dpi=300, bbox_inches='tight',
            )
        plt.close()


    def create_individual_plots(self, data: pd.DataFrame, data_detrended: pd.DataFrame,
                                time: np.ndarray, metric_config: dict):
        """Create individual plots for each experiment"""
        unique_experiments = data['Type'].unique()
        n_exp = len(unique_experiments)
        
        # Get color palette for consistent colors across both plots
        geno_palette = genotype_palette_for_data(data, 'Type')
        
        # Calculate y-axis limits for original data
        min_signal = data['Signal'].min()
        max_signal = data['Signal'].mean() + 3 * data['Signal'].std()
        
        # First Figure: Original Signals
        fig1 = plt.figure(figsize=(10, 3 * (n_exp + 1)), constrained_layout=True)
        gs1 = fig1.add_gridspec(n_exp + 1, 1)
        
        # Plot original data for each experiment
        for i, exp_name in enumerate(unique_experiments):
            ax = fig1.add_subplot(gs1[i, 0])
            exp_data = data[data['Type'] == exp_name]
            
            sns.lineplot(x="Time", y="Signal", data=exp_data,
                        errorbar='se', ax=ax, color=geno_palette.get(exp_name),
                        estimator=np.mean)
            tick_df = exp_data.rename(columns={'Time': 'ElapsedTime (h)'}) if 'ElapsedTime (h)' not in exp_data.columns else exp_data
            decorate_elapsed_time_axis(ax, tick_df, self.conf, day_axis=True)
            
            ax.set_ylabel(metric_config['ylabel'])
            ax.set_xlabel('')
            ax.set_ylim(min_signal, max_signal)
            ax.legend([f"{exp_name}"], loc='upper left')
        
        # Add reference sinusoids for original scale
        ax_sin = fig1.add_subplot(gs1[-1, 0])
        exp1 = -0.25 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = 0.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        ax_sin.plot(time, exp1, color='red', label='24h rhythm')
        ax_sin.plot(time, exp2, color='black', label='12h rhythm')
        self._clock_ticks(ax_sin, data, day_axis=True)
        
        ax_sin.set_ylabel('Reference Patterns')
        ax_sin.set_xlabel('Time (h)')
        ax_sin.legend(loc='upper right')
        
        plt.suptitle(f"{metric_config['title']} Analysis - Original", fontsize=16, y=1.02)
        
        # Save original plots
        for ext in ['png', 'svg']:
            plt.savefig(
                plot_file(self.conf, MODULE_TEMPORAL, self.slug, f'{self.slug}_individual_original.{ext}', 'growth_speed'),
                dpi=300, bbox_inches='tight',
            )
        plt.close()
        
        # Second Figure: Normalized Signals
        fig2 = plt.figure(figsize=(10, 3 * (n_exp + 1)), constrained_layout=True)
        gs2 = fig2.add_gridspec(n_exp + 1, 1)
        
        # Plot normalized data for each experiment
        for i, exp_name in enumerate(unique_experiments):
            ax = fig2.add_subplot(gs2[i, 0])
            exp_data_norm = data_detrended[data_detrended['Type'] == exp_name]
            
            sns.lineplot(x="Time", y="Signal", data=exp_data_norm,
                        errorbar='se', ax=ax, color=geno_palette.get(exp_name),
                        estimator=np.mean)
            self._clock_ticks(ax, exp_data_norm, day_axis=True)
            ax.set_ylabel(metric_config['norm_ylabel'])
            ax.set_xlabel('')
            ax.set_ylim(-1, 1)
            ax.legend([f"{exp_name}"], loc='upper left')
        
        # Add reference sinusoids for normalized scale
        ax_sin = fig2.add_subplot(gs2[-1, 0])
        exp1 = 0.25 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = -0.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        ax_sin.plot(time, exp1, color='red', label='24h rhythm')
        ax_sin.plot(time, exp2, color='black', label='12h rhythm')
        self._clock_ticks(ax_sin, data_detrended, day_axis=True)
        ax_sin.set_ylabel('Reference Patterns')
        ax_sin.set_xlabel('Time (h)')
        ax_sin.legend(loc='upper right')
        plt.suptitle(f"{metric_config['title']} Analysis - Normalized", fontsize=16, y=1.02)
        
        # Save normalized plots
        for ext in ['png', 'svg']:
            plt.savefig(
                plot_file(self.conf, MODULE_TEMPORAL, self.slug, f'{self.slug}_individual_normalized.{ext}', 'growth_speed'),
                dpi=300, bbox_inches='tight',
            )
        plt.close()

    def _plot_time_series(self, ax, data: pd.DataFrame, time: np.ndarray, 
                         ylabel: str, title: str):
        """Plot time series data"""
        geno_palette = genotype_palette_for_data(data, 'Type')
        geno_label = get_genotype_axis_label(self.conf)
        sns.lineplot(x="Time", y="Signal", data=data,
                    hue="Type", errorbar='se', ax=ax,
                    estimator=np.mean, palette=geno_palette)
        self._clock_ticks(ax, data, day_axis=True)
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (h)')
        ax.set_title(title, fontsize=16)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(geno_label)

    def _plot_fft(self, ax, data: pd.DataFrame, title: str):
        """Plot FFT data with annotations"""
        geno_palette = genotype_palette_for_data(data, 'Type')
        geno_label = get_genotype_axis_label(self.conf)
        sns.lineplot(x='Freqs', y='FFT', hue='Type',
                    data=data[data['Freqs'] >= 0], errorbar='se', ax=ax,
                    palette=geno_palette)
        
        # Add vertical lines for 24h and 12h periods
        periods = {'24h': 1/24, '12h': 1/12}
        colors = {'24h': 'red', '12h': 'black'}
        
        for period, freq in periods.items():
            # Find peak value at this frequency
            freq_data = data[np.abs(data['Freqs'] - freq) < 0.001]
            if not freq_data.empty:
                ax.axvline(x=freq, ymin=0, ymax=ax.get_ylim()[1],
                          color=colors[period], linestyle='--', alpha=0.5,
                          label=f'{period} period')
        
        ax.set_xlim(0, 0.5)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Frequency (1/hour)')
        ax.set_ylabel('Energy')
        
        # Ensure legend includes period markers
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', title=geno_label)

def makeFourierPlots(conf: dict):
    """Main function to create Fourier analysis plots"""
    try:
        processor = DataProcessor(conf)
        temporal_path = data_file(conf, 'Temporal_Data.csv')
        if not os.path.isfile(temporal_path):
            raise FileNotFoundError(f'Temporal_Data not found: {temporal_path}')
        temporal_df = pd.read_csv(temporal_path)

        for metric_type in MetricConfig.METRICS.keys():
            try:
                metric_config = MetricConfig.get_config(metric_type)
                if not _column_has_signal(temporal_df, metric_config['column']):
                    print(
                        f'Skipping Fourier metric {metric_type}: '
                        f'{metric_config["column"]} is missing or unmeasured',
                        flush=True,
                    )
                    continue
                visualizer = Visualizer(conf, metric_type)

                all_frames_original = processor.read_and_process_data(
                    temporal_df,
                    root=metric_config['column']
                )

                all_frames_detrended = processor.read_and_process_data(
                    temporal_df,
                    root=metric_config['column'],
                    normalize=True,
                    detrend=True,
                    medfilt=True
                )
                
                # Perform statistical analysis
                processor.perform_statistical_analysis(
                    all_frames_original, 
                    all_frames_detrended,
                    metric_type
                )
                
                # Create visualizations
                time_series = all_frames_original.loc[
                    all_frames_original['Time'].notna(), 'Time'
                ].drop_duplicates().to_numpy()
                visualizer.create_joint_plot(
                    all_frames_original,
                    all_frames_detrended,
                    time_series,
                    metric_config,
                    f"JointPlot_{metric_type}"
                )
                
                # Create individual plots - updated call to match function signature
                visualizer.create_individual_plots(
                    all_frames_original,
                    all_frames_detrended,
                    time_series,
                    metric_config
                )
                
            except Exception as e:
                print(f"Skipping metric {metric_type} due to error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in makeFourierPlots: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    conf = {
        'MainFolder': '/path/to/main/folder',
        'everyXhourFieldFourier': 24,
        'averagePerPlantStats': True
    }
    makeFourierPlots(conf)