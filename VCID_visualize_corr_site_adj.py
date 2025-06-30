import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress  # Import for Pearson correlation
import matplotlib.font_manager as fm
import graphviz

# Load data
base_path = '/Volumes/Alice_Data/VCID/'
corr_data_vascular_vs_obf = pd.read_excel(base_path + 'correlation_results_site_adj.xlsx', sheet_name='vascular_vs_obf_Combined')
corr_data_octa_vs_obf = pd.read_excel(base_path + 'correlation_results_site_adj.xlsx', sheet_name='octa_vs_obf_Combined')
corr_data_octa_vs_vascular = pd.read_excel(base_path + 'correlation_results_site_adj.xlsx', sheet_name='octa_vs_vascular_Combined')

# Load merged data (with adjusted columns)
merged_data_vascular = pd.read_excel(base_path + 'merged_data.xlsx', sheet_name='OBF_CBF_Vascular')
merged_data_octa = pd.read_excel(base_path + 'merged_data.xlsx', sheet_name='OBF_CBF_Vascular_OCTA')

# Visualization parameters
merged_data_octa['Source'] = merged_data_octa['Source'].replace({1: 'UPenn', 0: 'UMiami'})
merged_data_vascular['Source'] = merged_data_vascular['Source'].replace({1: 'UPenn', 0: 'UMiami'})

color_map = {'UPenn': '#1f77b4', 'UMiami': '#ff7f0e'}

def create_obf_vs_vascular_plots():
    # Define metrics for OBF vs Vascular comparison
    metrics_x = ['GM ACA+MCA', 'GM PCA']  # Vascular metrics (x-axis)
    hemispheres = ['ipsilateral', 'contralateral']
    rows = ['A', 'B']  # Updated to 'a' and 'b' only
    
    # Adjusted figure size for 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=False)  # Height reduced from 18 to 12
    for row_idx, hemisphere in enumerate(hemispheres):
        for col_idx, v_metric in enumerate(metrics_x):
            ax = axes[row_idx, col_idx]
            if hemisphere == 'ipsilateral':
                # Combine left and right data for ipsilateral using adjusted columns
                data_left = merged_data_vascular[['left_adj', f'Left {v_metric} _adj', 'Source']].copy()
                data_right = merged_data_vascular[['right_adj', f'Right {v_metric} _adj', 'Source']].copy()
                data_left.columns = ['rel_y', 'rel_x', 'Source']
                data_right.columns = ['rel_y', 'rel_x', 'Source']
                combined_data = pd.concat([data_left, data_right], ignore_index=True)
            else:  # contralateral
                # Combine left and right data for contralateral using adjusted columns
                data_left_contra = merged_data_vascular[['left_adj', f'Right {v_metric} _adj', 'Source']].copy()
                data_right_contra = merged_data_vascular[['right_adj', f'Left {v_metric} _adj', 'Source']].copy()
                data_left_contra.columns = ['rel_y', 'rel_x', 'Source']
                data_right_contra.columns = ['rel_y', 'rel_x', 'Source']
                combined_data = pd.concat([data_left_contra, data_right_contra], ignore_index=True)
            
            # Plot data points
            sns.scatterplot(
                data=combined_data,
                x='rel_x', y='rel_y',
                hue='Source', style='Source',
                palette=color_map,
                ax=ax
            )
            
            # Add regression line
            sns.regplot(
                data=combined_data,
                x='rel_x', y='rel_y',
                scatter=False,
                ax=ax,
                color='black'
            )
            
            # Add correlation info (combined across sites)
            corr_info = corr_data_vascular_vs_obf[
                (corr_data_vascular_vs_obf['X_Measure'] == v_metric) &
                (corr_data_vascular_vs_obf['Y_Measure'] == hemisphere)
            ]
            
            if not corr_info.empty:
                corr = corr_info.iloc[0]['Correlation']
                p_value = corr_info.iloc[0]['P_Value']
                if p_value < 0.01:
                    annotation = f"R={corr:.2f}, p<0.01*"
                else:
                    sig = '*' if p_value < 0.05 else ''
                    annotation = f"R={corr:.2f}, p={p_value:.2f}{sig}"
                ax.annotate(
                    annotation,
                    xy=(0.70, 0.92),
                    xycoords='axes fraction',
                    fontsize=13,
                    color='black'
                )
            
            ax.set_xlabel(f'{v_metric} (mL/100g/min)', fontsize=14, fontweight='bold')
            ax.set_ylabel('OBF (mL/100g/min)', fontsize=14, fontweight='bold')
            ax.set_ylim(15, 100)
            ax.set_xlim(23, 80)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Remove legend from individual plots
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    
    # Add a common legend at the bottom, adjusted for 2 rows
    handles, labels = ax.get_legend_handles_labels()
    fig.text(0.08, 0.89, rows[0], transform=fig.transFigure, fontsize=21, fontweight='bold', va='bottom', ha='left')
    fig.text(0.08, 0.91 - 0.45, rows[1], transform=fig.transFigure, fontsize=21, fontweight='bold', va='bottom', ha='left')
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.9), title='Site', title_fontsize=16, fontsize=14)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(base_path + 'OBF_vs_Vascular_correlation_plots_adj.tiff', dpi=300, bbox_inches='tight')
    plt.close()

def create_octa_vs_obf_plots():
    # Define metrics for OCTA vs OBF comparison
    metrics_y = ['RVN', 'DVP', 'SVP']  # OCTA metrics 
    hemispheres = ['ipsilateral', 'contralateral']
    rows = ['A', 'B']  # Updated to 'a' and 'b' only
    
    # Adjusted figure size for 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=False)  # Height reduced from 18 to 12
    for row_idx, hemisphere in enumerate(hemispheres):
        for col_idx, y_metric in enumerate(metrics_y):
            ax = axes[row_idx, col_idx]           
            if hemisphere == 'ipsilateral':
                # Combine left and right data for ipsilateral using adjusted columns
                data_left = merged_data_octa[['rel_left_adj', f'{y_metric}6_left_adj', 'Source']].copy()
                data_right = merged_data_octa[['rel_right_adj', f'{y_metric}6_right_adj', 'Source']].copy()
                data_left.columns = ['rel_x', 'rel_y', 'Source']
                data_right.columns = ['rel_x', 'rel_y', 'Source']
                combined_data = pd.concat([data_left, data_right], ignore_index=True)
                sns.scatterplot(
                    data=combined_data,
                    x='rel_x', y='rel_y',
                    hue='Source', style='Source',
                    palette=color_map,
                    ax=ax
                )
                sns.regplot(
                    data=combined_data,
                    x='rel_x', y='rel_y',
                    scatter=False,
                    ax=ax,
                    color='black'
                )
            else:  # contralateral
                # Combine left and right data for contralateral using adjusted columns
                data_left_contra = merged_data_octa[['rel_left_adj', f'{y_metric}6_right_adj', 'Source']].copy()
                data_right_contra = merged_data_octa[['rel_right_adj', f'{y_metric}6_left_adj', 'Source']].copy()
                data_left_contra.columns = ['rel_x', 'rel_y', 'Source']
                data_right_contra.columns = ['rel_x', 'rel_y', 'Source']
                combined_data_contra = pd.concat([data_left_contra, data_right_contra], ignore_index=True)
                sns.scatterplot(
                    data=combined_data_contra,
                    x='rel_x', y='rel_y',
                    hue='Source', style='Source',
                    palette=color_map,
                    ax=ax
                )
                sns.regplot(
                    data=combined_data_contra,
                    x='rel_x', y='rel_y',
                    scatter=False,
                    ax=ax,
                    color='black'
                )
            
            # Add correlation info (combined across sites)
            corr_info = corr_data_octa_vs_obf[
                (corr_data_octa_vs_obf['X_Measure'] == f'{hemisphere}') &
                (corr_data_octa_vs_obf['Y_Measure'] == f'{y_metric}')
            ]
            
            if not corr_info.empty:
                corr = corr_info.iloc[0]['Correlation']
                p_value = corr_info.iloc[0]['P_Value']
                if p_value < 0.01:
                    annotation = f"R={corr:.2f}, p<0.01*"
                else:
                    sig = '*' if p_value < 0.05 else ''
                    annotation = f"R={corr:.2f}, p={p_value:.2f}{sig}"
                if y_metric == 'DVP':
                    ax.annotate(
                        annotation,
                        xy=(0.67, 0.05),
                        xycoords='axes fraction',
                        fontsize=15,
                        color='black',
                    )
                else:
                    ax.annotate(
                        annotation,
                        xy=(0.67, 0.90),
                        xycoords='axes fraction',
                        fontsize=15,
                        color='black',
                    )
            # Set labels with larger, bold font
            ax.set_title(f'{y_metric}', fontsize=15, fontweight='bold')
            if col_idx == 1:  # Only set y-label for the first column
                ax.set_xlabel('Relative OBF', fontsize=20, fontweight='bold')
            else:
                ax.set_xlabel('')
            if col_idx == 0:  # Only set y-label for the first column
                ax.set_ylabel('Vessel Density (Dbox)', fontsize=20, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.set_ylim(1.812, 1.86)
            ax.set_xlim(0.5, 2.75)
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Remove legend from individual plots
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    
    # Add a common legend at the bottom, adjusted for 2 rows
    handles, labels = ax.get_legend_handles_labels()
    fig.text(0.08, 0.89, rows[0], transform=fig.transFigure, fontsize=25, fontweight='bold', va='bottom', ha='left')  
    fig.text(0.08, 0.89 - 0.43, rows[1], transform=fig.transFigure, fontsize=25, fontweight='bold', va='bottom', ha='left')  
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.9), title='Site', title_fontsize=16, fontsize=14)
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.savefig(base_path + 'OCTA_vs_OBF_correlation_plots_adj.tiff', dpi=300, bbox_inches='tight')
    plt.close()

def create_octa_vs_vascular_plots():
    # Define metrics for OCTA vs Vascular comparison
    octa_metrics = ['RVN', 'DVP', 'SVP']  # OCTA metrics
    vascular_metrics = ['gm_aca+mca', 'gm_pca']  # All vascular metrics
    vascular_labels = ['GM ACA+MCA', 'GM PCA']
    hemispheres = ['ipsilateral', 'contralateral']
    rows = ['A', 'B']  # Updated to 'a' and 'b' only
    
    # Create one plot for each vascular metric
    for v_idx, vascular_metric in enumerate(vascular_metrics):
        # Adjusted figure size for 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharey=False)  # Height reduced from 18 to 12
        
        for row_idx, hemisphere in enumerate(hemispheres):            
            for col_idx, octa_metric in enumerate(octa_metrics):
                ax = axes[row_idx, col_idx]
                
                if hemisphere == 'ipsilateral':
                    # Combine left and right data for ipsilateral using adjusted columns
                    data_left = merged_data_octa[[f'rel_left_{vascular_metric}_adj', f'{octa_metric}6_left_adj', 'Source']].copy()
                    data_right = merged_data_octa[[f'rel_right_{vascular_metric}_adj', f'{octa_metric}6_right_adj', 'Source']].copy()
                    data_left.columns = ['rel_x', 'rel_y', 'Source']
                    data_right.columns = ['rel_x', 'rel_y', 'Source']
                    combined_data = pd.concat([data_left, data_right], ignore_index=True)
                    sns.scatterplot(
                        data=combined_data,
                        x='rel_x', y='rel_y',
                        hue='Source', style='Source',
                        palette=color_map,
                        ax=ax
                    )
                    sns.regplot(
                        data=combined_data,
                        x='rel_x', y='rel_y',
                        scatter=False,
                        ax=ax,
                        color='black'
                    )
                else:  # contralateral
                    # Combine left and right data for contralateral using adjusted columns
                    data_left_contra = merged_data_octa[[f'rel_left_{vascular_metric}_adj', f'{octa_metric}6_right_adj', 'Source']].copy()
                    data_right_contra = merged_data_octa[[f'rel_right_{vascular_metric}_adj', f'{octa_metric}6_left_adj', 'Source']].copy()
                    data_left_contra.columns = ['rel_x', 'rel_y', 'Source']
                    data_right_contra.columns = ['rel_x', 'rel_y', 'Source']
                    combined_data_contra = pd.concat([data_left_contra, data_right_contra], ignore_index=True)
                    sns.scatterplot(
                        data=combined_data_contra,
                        x='rel_x', y='rel_y',
                        hue='Source', style='Source',
                        palette=color_map,
                        ax=ax
                    )
                    sns.regplot(
                        data=combined_data_contra,
                        x='rel_x', y='rel_y',
                        scatter=False,
                        ax=ax,
                        color='black'
                    )
                
                # Add correlation info (combined across sites)
                corr_info = corr_data_octa_vs_vascular[
                    (corr_data_octa_vs_vascular['X_Measure'] == f'{hemisphere}_{vascular_metric}') &
                    (corr_data_octa_vs_vascular['Y_Measure'] == f'{octa_metric}')
                ]
                
                if not corr_info.empty:
                    corr = corr_info.iloc[0]['Correlation']
                    p_value = corr_info.iloc[0]['P_Value']
                    if p_value < 0.01:
                        annotation = f"R={corr:.2f}, p<0.01*"
                    else:
                        sig = '*' if p_value < 0.05 else ''
                        annotation = f"R={corr:.2f}, p={p_value:.2f}{sig}"
                    if octa_metric == 'DVP':
                        ax.annotate(
                            annotation,
                            xy=(0.67, 0.05),
                            xycoords='axes fraction',
                            fontsize=15,
                            color='black',
                        )
                    else:
                        ax.annotate(
                            annotation,
                            xy=(0.67, 0.90),
                            xycoords='axes fraction',
                            fontsize=15,
                            color='black',
                        )
                # Set labels with larger, bold font
                if col_idx == 0:
                    ax.set_ylabel('Vessel Density (Dbox)', fontsize=20, fontweight='bold')
                else:
                    ax.set_ylabel('')
                if col_idx == 1:  # Only set x-label for the center subplot in each row
                    ax.set_xlabel(f'Relative {vascular_labels[v_idx]}', fontsize=20, fontweight='bold')
                else:
                    ax.set_xlabel('')
                    

                ax.set_title(f'{octa_metric}', fontsize=16, fontweight='bold')
                if vascular_metric == 'gm_aca+mca':
                    ax.set_ylim(1.812, 1.86)
                    ax.set_xlim(0.97, 1.72)
                elif vascular_metric == 'global_aca+mca':
                    ax.set_ylim(1.812, 1.86)
                    ax.set_xlim(0.8, 1.4)
                elif vascular_metric == 'gm_pca':
                    ax.set_ylim(1.812, 1.86)
                    ax.set_xlim(0.7, 1.5)
                elif vascular_metric == 'global_pca':
                    ax.set_ylim(1.812, 1.86)
                    ax.set_xlim(0.7, 1.4)
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Remove legend from individual plots
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        
        # Add a common legend at the bottom, adjusted for 2 rows
        handles, labels = ax.get_legend_handles_labels()
        fig.text(0.09, 0.9, rows[0], transform=fig.transFigure, fontsize=25, fontweight='bold', va='bottom', ha='left')  
        fig.text(0.09, 0.9 - 0.45, rows[1], transform=fig.transFigure, fontsize=25, fontweight='bold', va='bottom', ha='left')  
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.9), title='Site', title_fontsize=13, fontsize=12)
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        plt.savefig(base_path + f'OCTA_vs_{vascular_metric}_correlation_plots_adj.tiff', dpi=300, bbox_inches='tight')
        plt.close()

def combined_octa_vs_vascular_plots():
    # Define metrics for OCTA and comparisons
    octa_metrics = ['RVN', 'DVP', 'SVP']  # OCTA metrics
    comparisons = [
        ('rel_bilateral_adj', 'Relative OBF', 'corr_data_octa_vs_obf', 'rel_bilateral_adj'),  # First row: vs. OBF
        ('rel_bilateral_gm_aca+mca_adj', 'GM ACA+MCA', 'corr_data_octa_vs_vascular', 'rel_bilateral_gm_aca+mca_adj'),  # Second row: vs. GM ACA+MCA
        ('rel_bilateral_gm_pca_adj', 'GM PCA', 'corr_data_octa_vs_vascular', 'rel_bilateral_gm_pca_adj')  # Third row: vs. GM PCA
    ]
    rows = ['A', 'B', 'C']  # Labels for each row (unchanged as this function keeps 3 rows)

    # Create a 3x3 subplot grid (unchanged as this function compares different metrics, not hemispheres)
    # Convert target width to inches
    width_in = 170 / 25.4  # ≈ 6.85 in
    height_in = 175 / 25.4  # e.g., ≈ 4.7 in (adjust per content)
    fig, axes = plt.subplots(3, 3, figsize=(width_in, height_in), sharey=False)

    for row_idx, (x_metric, x_label, corr_data_source, corr_x_measure) in enumerate(comparisons):
        for col_idx, octa_metric in enumerate(octa_metrics):
            ax = axes[row_idx, col_idx]
            octa_col = f'{octa_metric}6_average_adj'

            # Plot scatterplot with data points
            sns.scatterplot(
                data=merged_data_octa,
                x=x_metric,
                y=octa_col,
                hue='Source',
                style='Source',
                palette=color_map,
                s=15,
                ax=ax
            )

            # Add regression line
            sns.regplot(
                data=merged_data_octa,
                x=x_metric,
                y=octa_col,
                scatter=False,
                ax=ax,
                line_kws={'linewidth': 1, 'color': 'black'}
            )

            # Add correlation info
            corr_data = corr_data_octa_vs_obf if corr_data_source == 'corr_data_octa_vs_obf' else corr_data_octa_vs_vascular
            corr_info = corr_data[
                (corr_data['X_Measure'] == corr_x_measure) &
                (corr_data['Y_Measure'] == f'{octa_metric}6_average_adj')
            ]

            if not corr_info.empty:
                corr = corr_info.iloc[0]['Correlation']
                p_value = corr_info.iloc[0]['P_Value']
                if p_value < 0.01:
                    annotation = f"r={corr:.2f}, p<0.01*"
                else:
                    sig = '*' if p_value < 0.05 else ''
                    annotation = f"r={corr:.2f}, p={p_value:.2f}{sig}"
                if octa_metric == 'DVP':
                    ax.annotate(
                        annotation,
                        xy=(0.45, 0.05),
                        xycoords='axes fraction',
                        fontsize=6,
                        color='black',
                    )
                else:
                    ax.annotate(
                        annotation,
                        xy=(0.45, 0.90),
                        xycoords='axes fraction',
                        fontsize=6,
                        color='black',
                    )

            ax.set_title(f'{octa_metric}', fontsize=8, fontweight='bold')

            # Set x and y labels
            if col_idx == 1:  # Only set x-label for the center subplot in each row
                if row_idx == 0:
                    ax.set_xlabel('Relative OBF', fontsize=8, fontweight='bold')
                elif row_idx == 1:
                    ax.set_xlabel('Relative GM ACA+MCA', fontsize=8, fontweight='bold')
                elif row_idx == 2:
                    ax.set_xlabel('Relative GM PCA', fontsize=8, fontweight='bold')
            else:
                ax.set_xlabel('')  # Empty x-label for non-center subplots
            if col_idx == 0:
                ax.set_ylabel('Vessel Density (Dbox)', fontsize=8, fontweight='bold')
            else:
                ax.set_ylabel('')

            # Set axis limits based on the metric
            ax.set_ylim(1.812, 1.86)
            ax.tick_params(axis='both', labelsize=6)  # or 9, 8, etc.

            if x_metric == 'rel_bilateral_adj':
                ax.set_xlim(0.5, 3.0)  # Limits for OBF
            elif x_metric == 'rel_bilateral_gm_aca+mca_adj':
                ax.set_xlim(1.1, 1.7)  # Limits for GM ACA+MCA
            elif x_metric == 'rel_bilateral_gm_pca_adj':
                ax.set_xlim(0.8, 1.6)  # Limits for GM PCA
            
            if row_idx == 0:
                ax.set_aspect(38, adjustable='box')
            elif row_idx == 1:
                ax.set_aspect(9, adjustable='box')
            else:
                ax.set_aspect(12, adjustable='box')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Remove legend from individual plots
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Add a common legend at the bottom (unchanged as this function keeps 3 rows)
    handles, labels = ax.get_legend_handles_labels()
    fig.text(0.06, 0.90, rows[0], transform=fig.transFigure, fontsize=10, fontweight='bold', va='bottom', ha='left')
    fig.text(0.06, 0.90 - 0.28, rows[1], transform=fig.transFigure, fontsize=10, fontweight='bold', va='bottom', ha='left')
    fig.text(0.06, 0.90 - 0.56, rows[2], transform=fig.transFigure, fontsize=10, fontweight='bold', va='bottom', ha='left')
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.8, 0.9), title='Site', title_fontsize=7, fontsize=6)

    # Adjust subplot spacing to add more space between rows and reduce space between columns
    plt.subplots_adjust(hspace=0.4, wspace=0.18)

    # Save the plot
    plt.savefig(base_path + 'Fig.5_OCTA_vs_OBF_Vascular_correlation_plots.tiff', dpi=300, format='tiff', bbox_inches='tight')
    plt.close()

def create_obf_vs_gm_cbf_plots():
    # Define metrics for OBF vs GM CBF comparison
    metrics_x = ['GM CBF', 'GM ACA+MCA', 'GM PCA']  # GM CBF metrics
    labels = ['A', 'B', 'C']  # Subplot labels

    # Create a 1x3 subplot grid with same figure size
    width_in = 170 / 25.4  # ≈ 6.85 in
    height_in = 48 / 25.4  # ≈ 6.89 in
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in), sharey=False)

    for col_idx, x_metric in enumerate(metrics_x):
        ax = axes[col_idx]

        # Plot data points for bilateral OBF vs. bilateral GM CBF
        rel_y = 'bilateral_adj'  # Adjusted absolute bilateral OBF
        rel_x = f'Bilateral {x_metric} _adj'  # Adjusted absolute bilateral GM CBF

        sns.scatterplot(
            data=merged_data_vascular,
            x=rel_x,
            y=rel_y,
            hue='Source',
            style='Source',
            palette=color_map,
            s=15,
            ax=ax
        )

        # Add regression line
        sns.regplot(
            data=merged_data_vascular,
            x=rel_x,
            y=rel_y,
            scatter=False,
            ax=ax,
            line_kws={'linewidth': 1, 'color': 'black'}
        )

        # Extract the data for correlation calculation
        x_data = merged_data_vascular[rel_x].dropna()
        y_data = merged_data_vascular[rel_y].dropna()

        # Align the data by dropping rows with NaN in either column
        paired_data = merged_data_vascular[[rel_x, rel_y]].dropna()
        x_data = paired_data[rel_x]
        y_data = paired_data[rel_y]

        # Calculate Pearson correlation coefficient and p-value
        if len(x_data) > 1 and len(y_data) > 1:  # Ensure there are enough data points
            corr, p_value = pearsonr(x_data, y_data)
        else:
            corr, p_value = np.nan, np.nan  # If not enough data, set to NaN

        # Add correlation info
        if not np.isnan(corr) and not np.isnan(p_value):
            if p_value < 0.01:
                annotation = f"r={corr:.2f}, p<0.01*"
            else:
                sig = '*' if p_value < 0.05 else ''
                annotation = f"r={corr:.2f}, p={p_value:.2f}{sig}"
            ax.annotate(
                annotation,
                xy=(0.50, 0.97),
                xycoords='axes fraction',
                fontsize=6,
                color='black'
            )

        # Set labels
        if col_idx == 0:
            ax.set_ylabel('OBF (mL/100g/min)', fontsize=7, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_xlabel(f'{x_metric} (mL/100g/min)', fontsize=7, fontweight='bold')

        # Set axis limits
        ax.set_xlim(20, 80)
        ax.set_ylim(20, 100)
        ax.set_yticks(np.arange(20, 101, 20))  # includes 100
        ax.set_xticks(np.arange(20, 81,10))   # includes 80

        ax.set_aspect(0.6)  # Set aspect ratio to 1:1 for better visualization

        # Set tick label size
        ax.tick_params(axis='both', labelsize=6)
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Remove legend from individual plots
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Add subplot labels (A, B, C)
    for col_idx, label in enumerate(labels):
        fig.text(0.07 + col_idx * 0.28, 0.98, label, transform=fig.transFigure, fontsize=8, fontweight='bold', va='bottom', ha='left')

    # Add a common legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.8, 0.9), title='Site', title_fontsize=7, fontsize=6)
    plt.subplots_adjust(wspace=0.15)

    plt.savefig(base_path + 'Fig.4_OBF_vs_GM_CBF_correlation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_obf_octa_vs_age_plots():
    # Create a 2x2 subplot grid with increased width
    width_in = 150 / 25.4  # ~7.87 in (increased from 6.85 in)
    height_in = 110 / 25.4  # ~3.15 in (adjusted for two rows)
    fig, ax = plt.subplots(2, 2, figsize=(width_in, height_in), sharey=False)

    plt.subplots_adjust(wspace=0.15, hspace=0.4)  # Match wspace from create_obf_vs_gm_cbf_plots, add hspace for rows

    # Plot 1: GM CBF vs Age (First row, first column)
    if all(col in merged_data_vascular.columns for col in ['Age at Enrollment', 'GM CBF ']):
        sns.regplot(
            x='Age at Enrollment', y='GM CBF ',
            data=merged_data_vascular,
            scatter_kws={'color': 'black', 'alpha': 0.8, 's': 5},
            line_kws={'color': 'black', 'linewidth': 1},
            ax=ax[0, 0]
        )
        slope, intercept, r_value, p_value, std_err = linregress(
            merged_data_vascular['Age at Enrollment'].dropna(),
            merged_data_vascular['GM CBF '].dropna()
        )
        if p_value < 0.01:
            annotation = f"r={r_value:.2f}, p<0.01*"
        else:
            sig = '*' if p_value < 0.05 else ''
            annotation = f"r={r_value:.2f}, p={p_value:.2f}{sig}"
        ax[0, 0].annotate(
            annotation,
            xy=(0.50, 0.97),
            xycoords='axes fraction',
            fontsize=6,
            color='black'
        )
        ax[0, 0].set_ylabel('GM CBF (mL/100g/min)', fontsize=8, fontweight='bold')
        ax[0, 0].set_ylim(20, 80)
        ax[0, 0].set_yticks(np.arange(20, 81, 20))
        print('cbf slope:', slope, 'p-value:', p_value)
    else:
        print("Skipping Plot 1: Missing required columns for GM CBF vs Age")

    # Plot 2: Bilateral OBF vs Age (First row, second column)
    if all(col in merged_data_vascular.columns for col in ['Age at Enrollment', 'bilateral']):
        sns.regplot(
            x='Age at Enrollment', y='bilateral',
            data=merged_data_vascular,
            scatter_kws={'color': 'black', 'alpha': 0.8, 's': 5},
            line_kws={'color': 'black', 'linewidth': 1},
            ax=ax[0, 1]
        )
        slope, intercept, r_value, p_value, std_err = linregress(
            merged_data_vascular['Age at Enrollment'].dropna(),
            merged_data_vascular['bilateral'].dropna()
        )
        if p_value < 0.01:
            annotation = f"r={r_value:.2f}, p<0.01*"
        else:
            sig = '*' if p_value < 0.05 else ''
            annotation = f"r={r_value:.2f}, p={p_value:.2f}{sig}"
        ax[0, 1].annotate(
            annotation,
            xy=(0.50, 0.97),
            xycoords='axes fraction',
            fontsize=6,
            color='black'
        )
        ax[0, 1].set_ylabel('OBF (mL/100g/min)', fontsize=8, fontweight='bold')
        ax[0, 1].set_ylim(20, 100)
        ax[0, 1].set_yticks(np.arange(20, 101, 20))
        print('obf slope:', slope, 'p-value:', p_value)
    else:
        print("Skipping Plot 2: Missing required columns for Bilateral OBF vs Age")

    # Plot 3: OCTA (RVN, DVP, SVP) vs Age (Second row, first column)
    y1 = ['RVN6_average', 'DVP6_average', 'SVP6_average']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['^', 'o', 'D']

    if 'Age at Enrollment' in merged_data_octa.columns and any(col in merged_data_octa.columns for col in y1):
        for i in range(3):
            if y1[i] in merged_data_octa.columns:
                ax[1, 0].scatter(
                    merged_data_octa['Age at Enrollment'], merged_data_octa[y1[i]],
                    color=colors[i], alpha=0.8, s=5, marker=markers[i], label=f'{y1[i]}'
                )
                sns.regplot(
                    x='Age at Enrollment', y=y1[i],
                    data=merged_data_octa,
                    scatter=False,
                    line_kws={'color': colors[i], 'linewidth': 1},
                    ax=ax[1, 0]
                )
                slope, intercept, r_value, p_value, std_err = linregress(
                    merged_data_octa['Age at Enrollment'].dropna(),
                    merged_data_octa[y1[i]].dropna()
                )
                print(f'{y1[i]} slope:', slope, 'p-value:', p_value)
                if p_value < 0.01:
                    annotation = f"r={r_value:.2f}, p<0.01*"
                else:
                    sig = '*' if p_value < 0.05 else ''
                    annotation = f"r={r_value:.2f}, p={p_value:.2f}{sig}"
                ax[1, 0].annotate(
                    annotation,
                    xy=(0.6, 1 - 0.07 * i),
                    xycoords='axes fraction',
                    fontsize=6,
                    color=colors[i]
                )
            else:
                print(f"Skipping {y1[i]}: Column not found in merged_data_octa")

        # Custom legend for OCTA plot
        labels = ['RVN', 'DVP', 'SVP']
        handles = [plt.Line2D([0], [0], marker=markers[i], color='w', label=label, markersize=5, markerfacecolor=color)
                   for i, (label, color) in enumerate(zip(labels, colors))]
        ax[1, 0].legend(title='OCTA', handles=handles, loc='lower left', frameon=True, bbox_to_anchor=(1, 0.8), title_fontsize=6, fontsize=5)
        ax[1, 0].set_ylabel('Vessel Density (Dbox)', fontsize=8, fontweight='bold')
        ax[1, 0].set_xlim(40, 81)
        ax[1, 0].set_ylim(1.812, 1.86)
    else:
        print("Skipping Plot 3: Missing required columns for OCTA vs Age")

    # Remove the unused subplot in the second row, second column
    fig.delaxes(ax[1, 1])

    # Set labels and limits for all subplots
    lists = ['A', 'B', 'C']
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0)]):
        ax[row, col].spines['top'].set_visible(False)
        ax[row, col].spines['right'].set_visible(False)
        ax[row, col].set_xlabel('Age (years)', fontsize=8, fontweight='bold')
        ax[0, col].set_xlim(20, 81)
        ax[0, col].set_xticks(np.arange(20, 81, 10))
        ax[row, col].tick_params(axis='both', labelsize=6) # Set aspect ratio to 1:1 for better visualization
        fig.text(-0.2, 1.15, lists[i], transform=ax[row, col].transAxes, fontsize=8, fontweight='bold', va='bottom', ha='center')
    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    # Save and display
    save_path = base_path + 'Fig.6_OBF_OCTA_vs_Age_correlation_plots'
    print(f"Attempting to save arbitrated plot to: {save_path}")
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    print("Plot saved successfully")
    plt.close()


# Generate all plots
# create_obf_vs_vascular_plots()
# create_octa_vs_obf_plots()
# create_octa_vs_vascular_plots()
# combined_octa_vs_vascular_plots()
create_obf_vs_gm_cbf_plots()
# create_obf_octa_vs_age_plots()