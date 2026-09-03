import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.optimize import curve_fit

class GerminationAnalyzer:
    """Analyzes seed germination data and generates plots."""
    
    def __init__(self, data=None, output_dir='.', dt=15, store_for_each_video=False, add_time_before_photo=0, time_cut=0):
        self.data = self._prepare_data(data)
        self.base_dir = output_dir
        self.plot_dirs = self._setup_directories()
        
        # Analysis parameters
        self.TIME_WINDOW = 1  # Convert minutes to hours
        self.ROLL_WINDOW = max(5, int(round(2.5 / self.TIME_WINDOW)))
        self.DETECTION_WINDOW = max(10, int(round(5 / self.TIME_WINDOW)))
        self.store_for_each_video = store_for_each_video
        self.add_time_before_photo = add_time_before_photo
        self.time_cut = time_cut
        
        self.germination_data = None
        self.statistics = None
    
    def _prepare_data(self, data):
        """Prepare input data."""
        if data is None:
            return None
        
        data = data.copy()
        data['Experiment'] = data['Experiment'].astype(str)
        data['Experiment'] = data['Experiment'].str.replace('/', '_')
        data['Plant_id'] = data['Plant_id'].str.replace('/', '_')
        data['Plant_id'] = data['Plant_id'].astype(str)
        
        return data[data['Experiment'] != 'Unknown']
    
    def _setup_directories(self):
        """Create output directories."""
        dirs = {
            'germination': os.path.join(self.base_dir, 'germination', 'GerminationPlots'),
            'germination_video': os.path.join(self.base_dir, 'germination', 'GerminationPlotsPerVideo'),
            'survival': os.path.join(self.base_dir, 'germination', 'KaplanMeierPlots'),
            'data': os.path.join(self.base_dir, 'germination', 'ProcessedData')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
        
    @staticmethod
    def hill_function(x, y0, max_germ, steepness, t50):
        """Four-parameter Hill function for curve fitting."""
        return y0 + (max_germ * x**steepness) / (t50**steepness + x**steepness)
    
    @staticmethod
    def germination_rate(x, y0, max_germ, steepness, t50):
        """Calculate instantaneous germination rate."""
        return (max_germ * steepness * t50**steepness * x**(steepness-1)) / \
               (t50**steepness + x**steepness)**2
    
    @staticmethod
    def find_TMGR(params):
        """Calculate Time of Maximum Germination Rate."""
        _, _, steepness, t50 = params
        return t50 * ((steepness - 1)/(steepness + 1))**(1/steepness)
    
    def analyze(self):
        """Run complete germination analysis."""
        self._process_raw_data()
        self._calculate_statistics()
        self._create_plots()
        
        timeline_df = self._export_plot_timeline() 
        
        return {
            'ProcessedData': self.data,
            'GerminationData': self.germination_data,
            'Statistics': self.statistics,
            'PlotTimeline': timeline_df      
        }
    
  
    def _detect_germination(self, group):
        """Detect germination events based on growth patterns."""
        group = group.copy()
        group = group.sort_values('ElapsedHours')
        group['GrowthRate'] = (group['Perim.']
                             .rolling(window=self.ROLL_WINDOW, min_periods=1)
                             .mean() - group['Perim.'].mean())
        
        group['RateChange'] = group['GrowthRate'].diff().fillna(0)
        group['ShortTermChange'] = group['GrowthRate'].diff(periods=self.DETECTION_WINDOW)
        group['LongTermChange'] = group['GrowthRate'].diff(periods=self.DETECTION_WINDOW * 2)
        
        group['TrendDirection'] = (group['RateChange']
                                 .apply(np.sign)
                                 .rolling(window=self.DETECTION_WINDOW, min_periods=1)
                                 .mean())
        
        group['Germinated'] = ((group['TrendDirection'] >= 0.8) & 
                              (group['ShortTermChange'] > 0.05) & 
                              (group['LongTermChange'] > 0.1))
        return group

    def _process_raw_data(self):
        """Process raw measurements to detect germination events."""
        data = self.data.copy()
        processed = data.groupby(data['Plant_id']).apply(self._detect_germination)

        if 'Plant_id' not in processed.columns:
            if 'Plant_id' in processed.index.names:
                processed = processed.reset_index(level='Plant_id')
            elif isinstance(processed.index, pd.MultiIndex):
                processed = processed.reset_index(level=0)
            else:
                processed = processed.reset_index() 
            if 'Plant_id' not in processed.columns:
                for candidate in ['level_0', 'index']:
                    if candidate in processed.columns:
                        processed = processed.rename(columns={candidate: 'Plant_id'})
                        break

        processed = processed.reset_index(drop=True)
        data = processed
        
        data['ElapsedHours'] = data['ElapsedHours'] + self.add_time_before_photo
        
        # Get unique combinations of Group and Video
        group_video_counts = (data[['Experiment', 'Video', 'SeedCount']]
                            .drop_duplicates()
                            .set_index(['Experiment', 'Video'])['SeedCount']
                            .to_dict())
        
        germinated_mask = data['Germinated'] == True
        germ_subset = data[germinated_mask].copy()

        # Summarize germination events
        germ_data = (germ_subset
                    .groupby(['Experiment', 'Plant_id', 'Video'])
                    .agg({
                        'ElapsedHours': 'min',
                        'Area': 'mean',
                        **({col: 'first' for col in ['PlateCondition', 'ExtraVariable']
                            if col in germ_subset.columns}),
                    })
                    .reset_index()
                    .rename(columns={
                        'ElapsedHours': 'GerminationTime',
                        'Area': 'SeedSize'
                    }))
                
        non_germ_entries = []
        non_germ_entries_for_data = []
        
        # Store UIDs that should be excluded (when we have too many seeds)
        uids_to_exclude = set()

        # For each Group-Video combination
        for (group, video), count in group_video_counts.items():
            # Get germinated seeds for this group and video
            germinated = germ_data[(germ_data['Experiment'] == group) & 
                                (germ_data['Video'] == video)]
            
            # Get all UIDs for this group and video from original data
            group_video_data = data[(data['Experiment'] == group) & (data['Video'] == video)]
            group_video_uids = set(group_video_data['Plant_id'].unique())
            
            # Get current UIDs for this group and video in germination data
            current_uids = set(germinated['Plant_id'])
            
            # Find missing UIDs for this group and video
            missing_uids = group_video_uids - current_uids
            
            if count > 0:  # If count is provided
                # Calculate how many missing UIDs we should add
                germinated_count = len(germinated)
                needed_count = count - germinated_count

                if needed_count < 0:
                    # We have MORE detected seeds than the expected count
                    # Sort by earliest germination time and keep only the first 'count' seeds
                    sorted_germinated = germinated.sort_values('GerminationTime')
                    keep_uids = set(sorted_germinated.head(count)['Plant_id'].values)
                    
                    # Find UIDs to exclude
                    exclude_uids = set(germinated['Plant_id']) - keep_uids
                    uids_to_exclude.update(exclude_uids)
                    continue  # Skip to next group-video combination
                
                if needed_count == 0:
                    continue  # Perfect match between detected and expected count

                # If we need more seeds than we have identified UIDs for
                missing_uids = list(missing_uids)[:needed_count]
                if len(missing_uids) < needed_count:
                    # Create additional entries with artificial UIDs
                    for i in range(needed_count - len(missing_uids)):
                        missing_uids.append(f"Group_{group}_video_{video}_NonGerm_{i}")

            # Add entries for missing plants
            for uid in missing_uids:
                factor_row = {
                    'Experiment': group,
                    'Video': video,
                    'Plant_id': uid,
                }
                if 'PlateCondition' in group_video_data.columns:
                    factor_row['PlateCondition'] = group_video_data['PlateCondition'].iloc[0]
                if 'ExtraVariable' in group_video_data.columns:
                    factor_row['ExtraVariable'] = group_video_data['ExtraVariable'].iloc[0]
                non_germ_entries.append({
                    **factor_row,
                    'GerminationTime': np.nan,
                    'SeedSize': np.nan
                })

                non_germ_entries_for_data.append({
                    'Experiment': group,
                    'Video': video,
                    'Plant_id': uid,
                    'ElapsedHours': 0,
                    'Perim.': np.nan,
                    'Area': np.nan,
                    'SeedCount': np.nan,
                    'GrowthRate': np.nan,
                    'RateChange': np.nan,
                    'ShortTermChange': np.nan,
                    'LongTermChange': np.nan,
                    'TrendDirection': np.nan,
                    'Germinated': False
                })

        # Filter out excess UIDs from germination data
        germ_data = germ_data[~germ_data['Plant_id'].isin(uids_to_exclude)]
        
        # Filter out excess UIDs from main data
        data = data[~data['Plant_id'].isin(uids_to_exclude)]

        # Add all non-germinated entries at once
        if non_germ_entries:
            germ_data = pd.concat([germ_data, pd.DataFrame(non_germ_entries)], 
                                ignore_index=True)
        
        if non_germ_entries_for_data:
            data = pd.concat([data, pd.DataFrame(non_germ_entries_for_data)], 
                        ignore_index=True)
        
        # Store the seed counts for each group-video in the data
        self.group_video_seed_counts = group_video_counts
        
        # Save processed data
        self.data = data
        self.germination_data = germ_data

        germ_data.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_Summary.csv'), 
                        sep=',', index=False)
        data.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_ProcessedData.csv'),
                    sep=',', index=False)
        
    def _calculate_statistics(self):
        """Calculate summary statistics for germination data."""
        if self.germination_data is None:
            raise ValueError("No germination data available. Run process_data first.")
        
        # Calculate group statistics
        stats = self.germination_data.groupby('Experiment').agg({
            'GerminationTime': ['count', 'mean', 'std'],
            'SeedSize': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = [
            'Experiment',
            'Germinated_Seeds',
            'Mean_Germination_Time',
            'SD_Germination_Time',
            'Mean_Seed_Size',
            'SD_Seed_Size'
        ]
        
        # Calculate ungerminated seeds
        ungerminated = (self.germination_data[self.germination_data['GerminationTime'].isna()]
                       .groupby('Experiment')
                       .size()
                       .reset_index()
                       .rename(columns={0: 'Ungerminated_Seeds'}))
        
        # Merge statistics
        self.statistics = stats.merge(ungerminated, on='Experiment', how='left')
        self.statistics.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_Statistics.csv'), 
                             sep=',', index=False)
        
        return self.statistics

    def _create_plots(self):
        """Generate all visualization plots."""
        # Original group plots
        for group_name, group_data in self.data.groupby('Experiment'):
            self._plot_germination_curve(group_name, group_data)
            
            # Add per-video plots for this group
            if self.store_for_each_video:
                for video_name, video_data in group_data.groupby('Video'):
                    self._plot_germination_curve_per_video(group_name, video_name, video_data)
        
        # Survival curves
        self._plot_survival_curves(pairwise=False)
        self._plot_survival_curves(pairwise=True)

    def _plot_germination_curve(self, group_name, group_data):
        """
        Create comprehensive germination curve plot using manual seed counts.
        """
        videos = group_data['Video'].unique()
    
        # Sum up the seed counts for all videos in this group
        total_seeds = sum(self.group_video_seed_counts.get((group_name, video), 0) 
                        for video in videos)
        
        # If there are no specified counts, fall back to counting UIDs
        if total_seeds == 0:
            total_seeds = len(group_data['Plant_id'].unique())
            
        time_points = sorted(group_data['ElapsedHours'].unique())
        germination_counts = []
        raw_counts = []  # Add list for raw counts
        
        for t in time_points:
            germinated = len(group_data[
                (group_data['ElapsedHours'] <= t) & 
                (group_data['Germinated'])
            ]['Plant_id'].unique())
            raw_counts.append(germinated)  # Store raw count
            germination_counts.append((germinated / total_seeds) * 100)
        
        try:
            # Fit Hill function
            popt, _ = curve_fit(
                self.hill_function, 
                time_points,
                germination_counts,
                p0=[0, max(germination_counts), 2, np.median(time_points)],
                bounds=([0, 0, 0, 0], [10, 100, 20, max(time_points)])
            )
            
            # Generate smooth curves
            x_smooth = np.linspace(0, max(time_points), 1000)
            y_smooth = self.hill_function(x_smooth, *popt)
            
            # Calculate metrics
            final_germ_percent = max(germination_counts)
            TMGR = self.find_TMGR(popt)
            final_germinated = len(group_data[group_data['Germinated']]['Plant_id'].unique())
            
            # Calculate T50s using fine grid
            x_fine = np.linspace(0, max(time_points), 10000)
            y_fine = self.hill_function(x_fine, *popt)
            
            # T50 calculations remain the same...
            idx_50_total = np.argmin(np.abs(y_fine - 50))
            t50_total = x_fine[idx_50_total]
            
            if final_germ_percent > 0:
                target_germinated = final_germ_percent / 2
                idx_50_germinated = np.argmin(np.abs(y_fine - target_germinated))
                t50_germinated = x_fine[idx_50_germinated]
            else:
                t50_germinated = None
            
            # Calculate RoG curve
            rog = [self.germination_rate(x, *popt) for x in x_smooth]
            
            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(7, 4))
            
            # Primary y-axis (percentage)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Germination Percentage (%)')
            
            # Secondary y-axis (counts)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Seeds')
            
            # Add detailed information to the title with proper spacing
            fig.suptitle(f'Germination Curve - {group_name}\n'
             f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)',
             fontsize=12, y=1.05)
            
            # Reference lines (percentage)
            ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            
            # Main curves
            scatter1 = ax1.scatter(time_points, germination_counts, alpha=0.5, 
                            color='black', label='Observed Data')
            line1 = ax1.plot(x_smooth, y_smooth, '-', color='blue', 
                        label='FPHF Fit')
            line2 = ax1.plot(x_smooth, rog, '--', color='green', 
                        label='Rate of Germination')
            
            # Time markers
            vline1 = ax1.axvline(x=TMGR, color='red', linestyle='--', 
                            label=f'TMGR: {TMGR:.1f}h')
            vline2 = ax1.axvline(x=t50_total, color='black', linestyle='--', 
                            label=f'T50 Total: {t50_total:.1f}h')
            if t50_germinated:
                vline3 = ax1.axvline(x=t50_germinated, color='gray', linestyle='--', 
                                label=f'T50 Germinated: {t50_germinated:.1f}h')
            
            # Set limits
            ax1.set_ylim(0, 100)
            ax2.set_ylim(0, total_seeds)
            ax1.set_xlim(0, min(max(time_points), self.time_cut) if self.time_cut > 0 else max(time_points))
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend for primary axis only
            ax1.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
            
            # Save plot
            plot_name = os.path.join(self.plot_dirs['germination'], 
                                f'GerminationCurve_{group_name}')
            plt.tight_layout()
            self._save_plot(plot_name)
                        
        except RuntimeError as e:
            print(f"Could not fit curve for group {group_name}: {str(e)}")

    def _plot_survival_curves(self, pairwise=False):
        """
        Create Kaplan-Meier survival curve plots.
        
        Args:
            pairwise: If True, creates separate plots for each pair of groups
        """
        # Prepare survival data
        data_surv = self.germination_data.copy()
        max_time = self.data['ElapsedHours'].max()
        
        data_surv['Germinated'] = ~data_surv['GerminationTime'].isna()
        data_surv['GerminationTime'] = data_surv['GerminationTime'].fillna(max_time)
        
        unique_groups = sorted(data_surv['Experiment'].unique())
        
        if not pairwise:
            # Plot all groups together
            plt.figure(figsize=(7, 4))
            
            for group in unique_groups:
                group_data = data_surv[data_surv['Experiment'] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['GerminationTime'], 
                       event_observed=group_data['Germinated'])
                
                # Plot germination curve (1 - survival)
                plt.step(kmf.survival_function_.index, 
                        1 - kmf.survival_function_, 
                        where='post', 
                        linewidth=2,
                        label=group)  # Use group name directly for legend
            
            plot_name = os.path.join(self.plot_dirs['survival'], 
                                   'KaplanMeier_AllGroups')
            self._format_survival_plot(max_time, 'Kaplan Meier - All Groups')
            self._save_plot(plot_name)
            
        else:
            # Create pairwise plots
            for i, group1 in enumerate(unique_groups):
                for j, group2 in enumerate(unique_groups[i+1:], i+1):
                    plt.figure(figsize=(7, 4))
                    
                    for group in [group1, group2]:
                        group_data = data_surv[data_surv['Experiment'] == group]
                        kmf = KaplanMeierFitter()
                        kmf.fit(group_data['GerminationTime'], 
                               event_observed=group_data['Germinated'])
                        
                        plt.step(kmf.survival_function_.index, 
                                1 - kmf.survival_function_, 
                                where='post', 
                                linewidth=2,
                                label=group)  # Use group name directly for legend
                    
                    plot_name = os.path.join(self.plot_dirs['survival'], 
                                           f'KaplanMeier_{group1}_vs_{group2}')
                    self._format_survival_plot(max_time, 
                                           f'Kaplan Meier - {group1} vs {group2}')
                    self._save_plot(plot_name)

    def _save_plot(self, name, formats=('.png', '.pdf')):
        """Save plot in multiple formats."""
        for fmt in formats:
            plt.savefig(name + fmt, bbox_inches='tight', dpi=300)
        plt.close()

    def _format_survival_plot(self, max_time, title):
        """Format Kaplan-Meier plot with standard elements."""
        plt.xlim(0, max_time)
        plt.ylim(0, 1)
        plt.xlabel('Time (hours)')
        plt.ylabel('Fraction of Germinated Seeds')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Format legend with better positioning and style
        plt.legend(title='Groups',
                  loc='center right',
                  frameon=True,
                  edgecolor='black',
                  fancybox=True,
                  shadow=True)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

    def _plot_germination_curve_per_video(self, group_name, video_name, video_data):
        """
        Create germination curve plot for a specific video.
        
        Args:
            group_name: Name of the treatment group
            video_name: Name/ID of the video
            video_data: DataFrame containing data for this video
        """
        # Get total seeds from stored seed counts
        total_seeds = self.group_video_seed_counts.get((group_name, video_name), 0)
        
        # If there's no specified count, fall back to counting UIDs
        if total_seeds == 0:
            total_seeds = len(video_data['Plant_id'].unique())

        time_points = sorted(video_data['ElapsedHours'].unique())
        germination_counts = []
        
        for t in time_points:
            germinated = len(video_data[
                (video_data['ElapsedHours'] <= t) & 
                (video_data['Germinated'])
            ]['Plant_id'].unique())
            germination_counts.append((germinated / total_seeds) * 100)
        
        try:
            # Fit Hill function
            popt, _ = curve_fit(
                self.hill_function, 
                time_points,
                germination_counts,
                p0=[0, max(germination_counts), 2, np.median(time_points)],
                bounds=([0, 0, 0, 0], [10, 100, 20, max(time_points)])
            )
            
            # Generate smooth curves
            x_smooth = np.linspace(0, max(time_points), 1000)
            y_smooth = self.hill_function(x_smooth, *popt)
            
            # Calculate metrics
            final_germ_percent = max(germination_counts)
            TMGR = self.find_TMGR(popt)
            final_germinated = len(video_data[video_data['Germinated']]['Plant_id'].unique())
            

            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(7, 4))
            
            # Add detailed information to the title
            plt.suptitle(f'Germination Curve - {group_name} - Video {video_name}', fontsize=14)
            plt.title(f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)', 
                    fontsize=10)

            # Primary y-axis (percentage)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Germination Percentage (%)')
            
            # Secondary y-axis (counts)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Seeds')
            
            # Add detailed information to the title
            plt.suptitle(f'Germination Curve - {group_name}', fontsize=14)
            plt.title(f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)',
                    fontsize=10)
            
            # Reference lines (percentage)
            ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            
            # Main curves
            scatter1 = ax1.scatter(time_points, germination_counts, alpha=0.5, 
                            color='black', label='Observed Data')
            line1 = ax1.plot(x_smooth, y_smooth, '-', color='blue', 
                        label='FPHF Fit')
            
            # Time markers
            vline1 = ax1.axvline(x=TMGR, color='red', linestyle='--', 
                            label=f'TMGR: {TMGR:.1f}h')
            
            # Set limits
            ax1.set_ylim(0, 100)
            ax2.set_ylim(0, total_seeds)
            ax1.set_xlim(0, min(max(time_points), self.time_cut) if self.time_cut > 0 else max(time_points))
                        
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend for primary axis only
            ax1.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
            
            # Save plot
            plot_name = os.path.join(self.plot_dirs['germination_video'], 
                                f'GerminationCurve_{group_name}_Video_{video_name}')
            self._save_plot(plot_name)
            
        except RuntimeError as e:
            print(f"Could not fit curve for group {group_name} video {video_name}: {str(e)}")

    def _export_plot_timeline(self):
        """
        Exports a detailed CSV timeline of germination counts and percentages
        per Group (Genotype/Variety) and Video to replicate plots externally.
        """
        records = []
        
        # Iterate through each genotype/variety
        for group_name, group_data in self.data.groupby('Experiment'):
            # Iterate through each video within that group
            for video_name, video_data in group_data.groupby('Video'):
                
                # Fetch expected total seeds or default to unique observed UIDs
                total_seeds = self.group_video_seed_counts.get((group_name, video_name), 0)
                if total_seeds == 0:
                    total_seeds = len(video_data['Plant_id'].unique())
                    
                # Get all unique timepoints for this specific video
                time_points = sorted(video_data['ElapsedHours'].unique())
                
                # Calculate counts and percentages for each timestep
                for t in time_points:
                    germinated_count = len(video_data[
                        (video_data['ElapsedHours'] <= t) & 
                        (video_data['Germinated'])
                    ]['Plant_id'].unique())
                    
                    percentage = (germinated_count / total_seeds * 100) if total_seeds > 0 else 0
                    
                    records.append({
                        'Group_Identifier': group_name,
                        'Video_ID': video_name,
                        'TimeStep_Hours': t,
                        'Total_Seeds': total_seeds,
                        'Germinated_Count': germinated_count,
                        'Germination_Percentage': percentage
                    })
        
        # Convert to DataFrame
        timeline_df = pd.DataFrame(records)
        
        # Sort for clean, chronological reading in CSV
        timeline_df = timeline_df.sort_values(by=['Group_Identifier', 'Video_ID', 'TimeStep_Hours'])
        
        # Save to the ProcessedData directory
        export_path = os.path.join(self.plot_dirs['data'], 'Germination_Plot_Replication.csv')
        timeline_df.to_csv(export_path, index=False)
        
        return timeline_df