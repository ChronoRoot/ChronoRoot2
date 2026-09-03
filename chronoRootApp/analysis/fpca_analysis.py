import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import MonomialBasis

import seaborn as sns
from scipy.stats import norm
from .stats_utils import perform_scalar_pairwise_stats, ensure_factor_columns
from .report_plots import emit_scalar_comparison_plots
from .utils.report_paths import (
    MODULE_TEMPORAL,
    analysis_dir,
    data_file,
    plot_file,
    temporal_metric_slug,
)
from .utils.report_style import (
    genotype_palette_for_data,
    get_genotype_axis_label,
)

plt.switch_backend('agg')

import logging
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)


DEFAULT_FPCA_COLUMNS = [
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'NumberOfLateralRoots',
    'MainOverTotal (%)',
    'DiscreteLateralDensity (LR/cm)',
    'HypocotylLength (mm)',
]


def performFPCA(conf_path):
    if isinstance(conf_path, dict):
        conf = conf_path
    else:
        with open(conf_path, 'r') as file:
            conf = json.load(file)

    COLUMNS = list(conf.get('fpcaMetrics') or DEFAULT_FPCA_COLUMNS)
    if not COLUMNS:
        print('FPCA skipped: no metrics selected')
        return

    basis = MonomialBasis
    inverse_rank_normalize = conf['normFPCA']
    number_of_components = int(conf['numComponentsFPCAField'])

    temporal_data_df = pd.read_csv(data_file(conf, 'Temporal_Data.csv'))
    temporal_data_df = ensure_factor_columns(temporal_data_df)
    temporal_data_df['Experiment'] = temporal_data_df['Experiment'].astype('str')
    temporal_data_df = temporal_data_df.sort_values(by='Experiment')
    temporal_data_df['Plant_id'] = (
        temporal_data_df['Plant_id'].astype('str') + ' (' + temporal_data_df['Experiment'] + ')'
    )
    COLUMNS = [col for col in COLUMNS if col in temporal_data_df.columns]
    if not COLUMNS:
        print('FPCA skipped: selected metrics are not in Temporal_Data.csv')
        return

    magnitudes_dict = {
        magnitude: temporal_data_df.pivot(
            columns='Plant_id', values=magnitude, index='ElapsedTime (h)'
        ).dropna()
        for magnitude in COLUMNS
    }
    get_expid = lambda plant_id: temporal_data_df.set_index('Plant_id')['Experiment'].to_dict()[plant_id]
    genotype_palette = genotype_palette_for_data(temporal_data_df)
    genotype_legend = get_genotype_axis_label(conf)

    plt.ioff()

    for magnitude in COLUMNS:
        mag_slug = temporal_metric_slug(magnitude)

        # Phase A: original 5x2 overview figure (verbatim layout)
        plt.figure(figsize=(8, 16))

        plt.subplot(5, 2, 1)
        sns.lineplot(
            x='ElapsedTime (h)', y=magnitude, hue='Experiment',
            data=temporal_data_df, errorbar='se', palette=genotype_palette,
        )
        plt.title(magnitude, fontsize=16)
        leg = plt.gca().get_legend()
        if leg is not None:
            leg.set_title(genotype_legend)

        fpca = FPCA(n_components=number_of_components, components_basis=basis)
        fpc_values = fpca.fit_transform(FDataGrid(magnitudes_dict[magnitude].transpose()))

        fpc_df = pd.DataFrame(fpc_values).set_index(magnitudes_dict[magnitude].columns)
        fpc_df.columns = [f'PC{i}' for i in range(1, fpca.n_components + 1)]
        fpc_df = fpc_df.reset_index()
        fpc_df['Experiment'] = fpc_df.Plant_id.apply(get_expid)
        plant_meta = temporal_data_df.drop_duplicates('Plant_id').set_index('Plant_id')
        for col in ['PlateCondition', 'ExtraVariable']:
            if col in plant_meta.columns:
                fpc_df[col] = fpc_df.Plant_id.map(plant_meta[col].to_dict())
        fpc_df = ensure_factor_columns(fpc_df)

        for j in range(1, fpca.n_components + 1):
            fpc_df[f'PC{j}_IRN'] = norm.ppf(fpc_df[f'PC{j}'].rank() / (len(fpc_df) + 1))
        fpc_df = fpc_df.sort_values(by='Experiment')

        ax = plt.subplot(5, 2, 2)
        sns.scatterplot(
            data=fpc_df,
            x='PC1' + ('_IRN' if inverse_rank_normalize else ''),
            y='PC2' + ('_IRN' if inverse_rank_normalize else ''),
            hue='Experiment',
            palette=genotype_palette,
            s=100,
            ax=ax,
        )
        ax.set_title('PC1 vs PC2')
        ax.set_xlabel('PC1' + (' (IRN)' if inverse_rank_normalize else ''))
        ax.set_ylabel('PC2' + (' (IRN)' if inverse_rank_normalize else ''))
        ax.legend(title=genotype_legend, bbox_to_anchor=(1.05, 1), loc='upper left')

        for fpc1 in range(1, number_of_components + 1):
            ax = plt.subplot(5, 2, 1 + fpc1 * 2)
            sns.boxplot(
                data=fpc_df,
                x='Experiment',
                hue='Experiment',
                y=f"PC{fpc1}{'_IRN' if inverse_rank_normalize else ''}",
                ax=ax,
                palette=genotype_palette,
            )
            ax.set_title(
                f'PC{fpc1}. Variance Explained: {fpca.explained_variance_ratio_[fpc1 - 1]:.2f}',
                fontsize=16,
            )

            ax = plt.subplot(5, 2, 1 + fpc1 * 2 + 1)
            N = 10
            quantiles = np.arange(N + 1) / N
            z_quantiles = np.quantile(fpc_values, quantiles, axis=0)[1:-1]

            N = 8
            palette = sns.color_palette('coolwarm', N + 1)

            for i in range(z_quantiles.shape[0]):
                z_value = z_quantiles[i, fpc1 - 1]
                z = z_value * np.identity(number_of_components)[:, (fpc1 - 1)]
                curve = fpca.inverse_transform(z)
                curve = [x[0] for x in curve.data_matrix[0]]
                color = palette[i]
                ax.plot(curve, color=color, label=f'Q {quantiles[i]:.2f}')

            ax.set_title(f'Interpretation of PC{fpc1}', fontsize=16)
            ax.set_ylabel(magnitude)
            ax.set_xlabel('Time (h)')

            handles = [plt.Line2D([0, 1], [0, 1], color=palette[i], lw=2) for i in range(N + 1)][::-1]
            labels = [f'{z_quantiles[i, fpc1 - 1]:.2f}' for i in range(N + 1)][::-1]
            ax.legend(handles, labels, title=f'PC{fpc1} Value', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(
                plot_file(conf, MODULE_TEMPORAL, mag_slug, f'{mag_slug}_overview.{ext}', 'fpca'),
                dpi=300, bbox_inches='tight',
            )
        plt.close()
        plt.cla()
        plt.clf()

        if number_of_components >= 2:
            for i in range(1, number_of_components + 1):
                for j in range(i + 1, number_of_components + 1):
                    plt.figure(figsize=(8, 6))
                    fpc_i = f"PC{i}{'_IRN' if inverse_rank_normalize else ''}"
                    fpc_j = f"PC{j}{'_IRN' if inverse_rank_normalize else ''}"
                    sns.scatterplot(
                        data=fpc_df, x=fpc_i, y=fpc_j, hue='Experiment',
                        palette=genotype_palette, s=100,
                    )
                    plt.title(f'{magnitude} - PC{i} vs PC{j}', fontsize=14)
                    plt.xlabel(f'PC{i}', fontsize=12)
                    plt.ylabel(f'PC{j}', fontsize=12)
                    plt.legend(title=genotype_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    for ext in ('png', 'svg'):
                        plt.savefig(
                            plot_file(
                                conf, MODULE_TEMPORAL, mag_slug,
                                f'{mag_slug}_pc{i}_vs_pc{j}.{ext}', 'fpca',
                            ),
                            dpi=300, bbox_inches='tight',
                        )
                    plt.close()
                    plt.cla()
                    plt.clf()

        # Phase B: paired comparison stats/plots per PC (after overview is saved)
        for fpc1 in range(1, number_of_components + 1):
            pc_col = f"PC{fpc1}{'_IRN' if inverse_rank_normalize else ''}"
            pc_label = f'{magnitude} — PC{fpc1}'
            pc_dir = analysis_dir(conf, MODULE_TEMPORAL, mag_slug, 'fpca', f'pc{fpc1}')
            perform_scalar_pairwise_stats(
                conf, fpc_df, pc_col, output_dir=None, plant_id_col='Plant_id',
                module=MODULE_TEMPORAL, metric_slug_name=mag_slug,
                subpath=('fpca', f'pc{fpc1}'), analysis_type='fpca',
            )
            emit_scalar_comparison_plots(
                conf, fpc_df, pc_col, pc_dir,
                module=MODULE_TEMPORAL, metric_slug_name=mag_slug,
                analysis_type='fpca', metric_label=pc_label,
            )
