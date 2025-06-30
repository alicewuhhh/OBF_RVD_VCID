## import libraries
import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Type aliases for compatibility
np.float = float
np.bool = bool
np.int = int
np.object = object

def load_data():
    """Load data from Excel files."""
    file_paths = {
        'demographics': '/Volumes/Alice_Data/VCID/VCID-demographics-cognition-Penn-UMiami-4-27-2023(14522).xlsx',
        'octa': '/Volumes/Alice_Data/VCID/OCTA_alldata_wide.xlsx',
        'cbf': '/Volumes/Alice_Data/VCID/VCID_CBF_aim1+2_combine.xlsx',
        'obf': '/Volumes/Alice_Data/VCID/VCID_OBF_aim1+2_combine.xlsx',
        'vascular': '/Volumes/Alice_Data/VCID/VCID_Vascular_aim1+2_combine.xlsx'
    }
    
    dmg = pd.read_excel(file_paths['demographics'], sheet_name='VCID subjects', usecols=range(3))
    octa = pd.read_excel(file_paths['octa']) if os.path.exists(file_paths['octa']) else None
    cbf = pd.read_excel(file_paths['cbf'], sheet_name='Sheet2')
    obf = pd.read_excel(file_paths['obf'], sheet_name='all_leftrightbi')
    vascular = pd.read_excel(file_paths['vascular'], sheet_name='Sheet1')
    
    for df in [cbf, vascular]:
        df.columns = df.columns.str.strip()
    return dmg, octa, cbf, obf, vascular

def preprocess_cbf(df):
    """Preprocess CBF data."""
    for i in range(6):
        col1, col2 = df.columns[i+1], df.columns[i+8]
        new_col = col1 + ' '
        df[new_col] = np.where(
            (df['QEI'] < 0.5) & (df['QEI.1'] >= 0.5), df[col2],
            np.where((df['QEI.1'] < 0.5) & (df['QEI'] >= 0.5), df[col1],
                    df[[col1, col2]].mean(axis=1))
        ).astype('float64')
    
    df = df[['SUBID', 'Global CBF ', 'GM CBF ', 'WM CBF ', 'PVWM12.5 CBF ', 
             'PVWM15 CBF ', 'Putamen CBF ', 'QEI', 'QEI.1']]
    df.loc[:, 'rel GM CBF '] = df['GM CBF '] / df['Putamen CBF ']  # Fixed SettingWithCopyWarning
    df = df.loc[(df['QEI'] >= 0.5) | (df['QEI.1'] >= 0.5)].copy()
    return df

def preprocess_octa(df):
    """Preprocess OCTA data."""
    df = df[['SUBID', 'DVP6_right', 'RVN6_right', 'SVP6_right',
             'DVP6_left', 'RVN6_left', 'SVP6_left',
             'DVP6_average', 'RVN6_average', 'SVP6_average']]
    # print("Columns after preprocess_octa:", df.columns.tolist())  # Debug
    return df

def preprocess_vascular(df, cbf_df):
    """Preprocess vascular territory data."""
    df = df.copy()
    for i in range(8):
        col1, col2 = df.columns[i+1], df.columns[i+9]
        df[col1 + ' '] = df[[col1, col2]].mean(axis=1)
    df = df[['SUBID', 'Left Global ACA+MCA ', 'Left Global PCA ', 'Left GM ACA+MCA ', 'Left GM PCA ',
             'Right Global ACA+MCA ', 'Right Global PCA ', 'Right GM ACA+MCA ', 'Right GM PCA ']]
    df = pd.merge(df, cbf_df[['SUBID', 'Putamen CBF ']], on='SUBID', how='inner')

    for side in ['Left', 'Right']:
        for measure in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']:
            abs_col = f'{side} {measure} '
            rel_col = f'rel_{side.lower()}_{measure.lower().replace(" ", "_")}'
            if abs_col in df.columns:
                df[rel_col] = df[abs_col] / df['Putamen CBF ']
    
    # Add bilateral measures
    for measure in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']:
        left_col = f'Left {measure} '
        right_col = f'Right {measure} '
        bilateral_col = f'Bilateral {measure} '
        if left_col in df.columns and right_col in df.columns:
            df[bilateral_col] = df[[left_col, right_col]].mean(axis=1)
        df[f'rel_bilateral_{measure.lower().replace(" ", "_")}'] = df[[f'rel_left_{measure.lower().replace(" ", "_")}', f'rel_right_{measure.lower().replace(" ", "_")}']].mean(axis=1)
    return df.dropna()

def preprocess_obf(df, cbf_df):
    """Preprocess OBF data."""
    df = df.copy()
    df = df[['SUBID', 'left', 'right', 'bilateral', 'Source']]
    df.loc[:, 'Source'] = df['Source'].replace({'Miami': 0, 'Penn': 1})
    df = pd.merge(df, cbf_df[['SUBID', 'Putamen CBF ']], on='SUBID', how='inner')
    df['rel_left'] = df['left'] / df['Putamen CBF ']
    df['rel_right'] = df['right'] / df['Putamen CBF ']
    df['rel_bilateral'] = df[['left', 'right']].mean(axis=1) / df['Putamen CBF ']
    # print("Columns after preprocess_obf:", df.columns.tolist())  # Debug
    return df

def correct_site_effect(df, value_cols, source_col='Source'):
    """Correct site effect using OLS regression."""
    df = df.copy()
    for col in value_cols:
        # print(f"Correcting site effect for column: {col}")  # Debug
        if col in df.columns and not df[col].isna().all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            X = sm.add_constant(df[[source_col]].astype(float))
            mask = df[col].notna()
            if mask.sum() > 1:
                try:
                    model = sm.OLS(df.loc[mask, col], X.loc[mask]).fit()
                    source_mean = df[source_col].mean()
                    df.loc[mask, f'{col}_adj'] = df.loc[mask, col] - (model.params[source_col] * (df.loc[mask, source_col] - source_mean))
                except Exception as e:
                    print(f"Warning: Could not fit model for {col}: {str(e)}")
    return df

def merge_data(dmg, cbf_df, obf_df, vascular_df, octa_df=None):
    """Merge datasets."""
    df = pd.merge(dmg, cbf_df, on='SUBID', how='inner')
    df = pd.merge(df, obf_df, on='SUBID', how='inner')
    df = pd.merge(df, vascular_df, on='SUBID', how='inner')
    if octa_df is not None:
        df = pd.merge(df, octa_df, on='SUBID', how='inner')
    # print("Shape of merged df:", df.shape)  # Debug
    return df


def generate_corr_pairs():
    """Generate correlation pairs for all comparisons (non-adjusted)."""
    pairs = {
        'octa_vs_vascular': {
            'ipsilateral': [
                ('rel_left_global_aca+mca', 'RVN6_left'), ('rel_right_global_aca+mca', 'RVN6_right'),
                ('rel_left_global_aca+mca', 'DVP6_left'), ('rel_right_global_aca+mca', 'DVP6_right'),
                ('rel_left_global_aca+mca', 'SVP6_left'), ('rel_right_global_aca+mca', 'SVP6_right'),
                ('rel_left_global_pca', 'RVN6_left'), ('rel_right_global_pca', 'RVN6_right'),
                ('rel_left_global_pca', 'DVP6_left'), ('rel_right_global_pca', 'DVP6_right'),
                ('rel_left_global_pca', 'SVP6_left'), ('rel_right_global_pca', 'SVP6_right'),
                ('rel_left_gm_aca+mca', 'RVN6_left'), ('rel_right_gm_aca+mca', 'RVN6_right'),
                ('rel_left_gm_aca+mca', 'DVP6_left'), ('rel_right_gm_aca+mca', 'DVP6_right'),
                ('rel_left_gm_aca+mca', 'SVP6_left'), ('rel_right_gm_aca+mca', 'SVP6_right'),
                ('rel_left_gm_pca', 'RVN6_left'), ('rel_right_gm_pca', 'RVN6_right'),
                ('rel_left_gm_pca', 'DVP6_left'), ('rel_right_gm_pca', 'DVP6_right'),
                ('rel_left_gm_pca', 'SVP6_left'), ('rel_right_gm_pca', 'SVP6_right')
            ],
            'contralateral': [
                ('rel_left_global_aca+mca', 'RVN6_right'), ('rel_right_global_aca+mca', 'RVN6_left'),
                ('rel_left_global_aca+mca', 'DVP6_right'), ('rel_right_global_aca+mca', 'DVP6_left'),
                ('rel_left_global_aca+mca', 'SVP6_right'), ('rel_right_global_aca+mca', 'SVP6_left'),
                ('rel_left_global_pca', 'RVN6_right'), ('rel_right_global_pca', 'RVN6_left'),
                ('rel_left_global_pca', 'DVP6_right'), ('rel_right_global_pca', 'DVP6_left'),
                ('rel_left_global_pca', 'SVP6_right'), ('rel_right_global_pca', 'SVP6_left'),
                ('rel_left_gm_aca+mca', 'RVN6_right'), ('rel_right_gm_aca+mca', 'RVN6_left'),
                ('rel_left_gm_aca+mca', 'DVP6_right'), ('rel_right_gm_aca+mca', 'DVP6_left'),
                ('rel_left_gm_aca+mca', 'SVP6_right'), ('rel_right_gm_aca+mca', 'SVP6_left'),
                ('rel_left_gm_pca', 'RVN6_right'), ('rel_right_gm_pca', 'RVN6_left'),
                ('rel_left_gm_pca', 'DVP6_right'), ('rel_right_gm_pca', 'DVP6_left'),
                ('rel_left_gm_pca', 'SVP6_right'), ('rel_right_gm_pca', 'SVP6_left')
            ],
            'bilateral': [
                ('rel_bilateral_global_aca+mca', 'RVN6_average'), ('rel_bilateral_global_aca+mca', 'DVP6_average'),
                ('rel_bilateral_global_aca+mca', 'SVP6_average'), ('rel_bilateral_global_pca', 'RVN6_average'),
                ('rel_bilateral_global_pca', 'DVP6_average'), ('rel_bilateral_global_pca', 'SVP6_average'),
                ('rel_bilateral_gm_aca+mca', 'RVN6_average'), ('rel_bilateral_gm_aca+mca', 'DVP6_average'),
                ('rel_bilateral_gm_aca+mca', 'SVP6_average'), ('rel_bilateral_gm_pca', 'RVN6_average'),
                ('rel_bilateral_gm_pca', 'DVP6_average'), ('rel_bilateral_gm_pca', 'SVP6_average')
            ]
        },
        'octa_vs_obf': {
            'ipsilateral': [
                ('rel_left', 'RVN6_left'), ('rel_right', 'RVN6_right'),
                ('rel_left', 'DVP6_left'), ('rel_right', 'DVP6_right'),
                ('rel_left', 'SVP6_left'), ('rel_right', 'SVP6_right')
            ],
            'contralateral': [
                ('rel_left', 'RVN6_right'), ('rel_right', 'RVN6_left'),
                ('rel_left', 'DVP6_right'), ('rel_right', 'DVP6_left'),
                ('rel_left', 'SVP6_right'), ('rel_right', 'SVP6_left')
            ],
            'bilateral': [
                ('rel_bilateral', 'RVN6_average'), ('rel_bilateral', 'DVP6_average'),
                ('rel_bilateral', 'SVP6_average')
            ]
        },
        'vascular_vs_obf': {
            'ipsilateral': [
                ('Left Global ACA+MCA ', 'left'), ('Right Global ACA+MCA ', 'right'),
                ('Left Global PCA ', 'left'), ('Right Global PCA ', 'right'),
                ('Left GM ACA+MCA ', 'left'), ('Right GM ACA+MCA ', 'right'),
                ('Left GM PCA ', 'left'), ('Right GM PCA ', 'right')
            ],
            'contralateral': [
                ('Right Global ACA+MCA ', 'left'), ('Left Global ACA+MCA ', 'right'),
                ('Right Global PCA ', 'left'), ('Left Global PCA ', 'right'),
                ('Right GM ACA+MCA ', 'left'), ('Left GM ACA+MCA ', 'right'),
                ('Right GM PCA ', 'left'), ('Left GM PCA ', 'right')
            ],
            'bilateral': [
                ('Bilateral Global ACA+MCA ', 'bilateral'), ('Bilateral Global PCA ', 'bilateral'),
                ('Bilateral GM ACA+MCA ', 'bilateral'), ('Bilateral GM PCA ', 'bilateral')
            ]
        }
    }
    return pairs

def generate_corr_pairs_adj():
    """Generate correlation pairs using adjusted columns."""
    pairs = {
        'octa_vs_vascular': {
            'ipsilateral-left': [
                ('rel_left_global_aca+mca_adj', 'RVN6_left_adj'), ('rel_left_global_aca+mca_adj', 'DVP6_left_adj'),
                ('rel_left_global_aca+mca_adj', 'SVP6_left_adj'), ('rel_left_global_pca_adj', 'RVN6_left_adj'),
                ('rel_left_global_pca_adj', 'DVP6_left_adj'), ('rel_left_global_pca_adj', 'SVP6_left_adj'),
                ('rel_left_gm_aca+mca_adj', 'RVN6_left_adj'), ('rel_left_gm_aca+mca_adj', 'DVP6_left_adj'),
                ('rel_left_gm_aca+mca_adj', 'SVP6_left_adj'), ('rel_left_gm_pca_adj', 'RVN6_left_adj'),
                ('rel_left_gm_pca_adj', 'DVP6_left_adj'), ('rel_left_gm_pca_adj', 'SVP6_left_adj')
            ],
            'ipsilateral-right': [
                ('rel_right_global_aca+mca_adj', 'RVN6_right_adj'), ('rel_right_global_aca+mca_adj', 'DVP6_right_adj'),
                ('rel_right_global_aca+mca_adj', 'SVP6_right_adj'), ('rel_right_global_pca_adj', 'RVN6_right_adj'),
                ('rel_right_global_pca_adj', 'DVP6_right_adj'), ('rel_right_global_pca_adj', 'SVP6_right_adj'),
                ('rel_right_gm_aca+mca_adj', 'RVN6_right_adj'), ('rel_right_gm_aca+mca_adj', 'DVP6_right_adj'),
                ('rel_right_gm_aca+mca_adj', 'SVP6_right_adj'), ('rel_right_gm_pca_adj', 'RVN6_right_adj'),
                ('rel_right_gm_pca_adj', 'DVP6_right_adj'), ('rel_right_gm_pca_adj', 'SVP6_right_adj')
            ],
            'contralateral-left': [
                ('rel_left_global_aca+mca_adj', 'RVN6_right_adj'), ('rel_left_global_aca+mca_adj', 'DVP6_right_adj'),
                ('rel_left_global_aca+mca_adj', 'SVP6_right_adj'), ('rel_left_global_pca_adj', 'RVN6_right_adj'),
                ('rel_left_global_pca_adj', 'DVP6_right_adj'), ('rel_left_global_pca_adj', 'SVP6_right_adj'),
                ('rel_left_gm_aca+mca_adj', 'RVN6_right_adj'), ('rel_left_gm_aca+mca_adj', 'DVP6_right_adj'),
                ('rel_left_gm_aca+mca_adj', 'SVP6_right_adj'), ('rel_left_gm_pca_adj', 'RVN6_right_adj'),
                ('rel_left_gm_pca_adj', 'DVP6_right_adj'), ('rel_left_gm_pca_adj', 'SVP6_right_adj')
            ],
            'contralateral-right': [
                ('rel_right_global_aca+mca_adj', 'RVN6_left_adj'), ('rel_right_global_aca+mca_adj', 'DVP6_left_adj'),
                ('rel_right_global_aca+mca_adj', 'SVP6_left_adj'), ('rel_right_global_pca_adj', 'RVN6_left_adj'),
                ('rel_right_global_pca_adj', 'DVP6_left_adj'), ('rel_right_global_pca_adj', 'SVP6_left_adj'),
                ('rel_right_gm_aca+mca_adj', 'RVN6_left_adj'), ('rel_right_gm_aca+mca_adj', 'DVP6_left_adj'),
                ('rel_right_gm_aca+mca_adj', 'SVP6_left_adj'), ('rel_right_gm_pca_adj', 'RVN6_left_adj'),
                ('rel_right_gm_pca_adj', 'DVP6_left_adj'), ('rel_right_gm_pca_adj', 'SVP6_left_adj')
            ],
            'bilateral': [
                ('rel_bilateral_global_aca+mca_adj', 'RVN6_average_adj'), ('rel_bilateral_global_aca+mca_adj', 'DVP6_average_adj'),
                ('rel_bilateral_global_aca+mca_adj', 'SVP6_average_adj'), ('rel_bilateral_global_pca_adj', 'RVN6_average_adj'),
                ('rel_bilateral_global_pca_adj', 'DVP6_average_adj'), ('rel_bilateral_global_pca_adj', 'SVP6_average_adj'),
                ('rel_bilateral_gm_aca+mca_adj', 'RVN6_average_adj'), ('rel_bilateral_gm_aca+mca_adj', 'DVP6_average_adj'),
                ('rel_bilateral_gm_aca+mca_adj', 'SVP6_average_adj'), ('rel_bilateral_gm_pca_adj', 'RVN6_average_adj'),
                ('rel_bilateral_gm_pca_adj', 'DVP6_average_adj'), ('rel_bilateral_gm_pca_adj', 'SVP6_average_adj')
            ]
        },
        'octa_vs_obf': {
            'ipsilateral-left': [
                ('rel_left_adj', 'RVN6_left_adj'), ('rel_left_adj', 'DVP6_left_adj'), ('rel_left_adj', 'SVP6_left_adj')
            ],
            'ipsilateral-right': [
                ('rel_right_adj', 'RVN6_right_adj'), ('rel_right_adj', 'DVP6_right_adj'), ('rel_right_adj', 'SVP6_right_adj')
            ],
            'contralateral-left': [
                ('rel_left_adj', 'RVN6_right_adj'), ('rel_left_adj', 'DVP6_right_adj'), ('rel_left_adj', 'SVP6_right_adj')
            ],
            'contralateral-right': [
                ('rel_right_adj', 'RVN6_left_adj'), ('rel_right_adj', 'DVP6_left_adj'), ('rel_right_adj', 'SVP6_left_adj')
            ],
            'bilateral': [
                ('rel_bilateral_adj', 'RVN6_average_adj'), ('rel_bilateral_adj', 'DVP6_average_adj'),
                ('rel_bilateral_adj', 'SVP6_average_adj')
            ]
        },
        'vascular_vs_obf': {
            'ipsilateral-left': [
                ('Left Global ACA+MCA _adj', 'left_adj'), ('Left Global PCA _adj', 'left_adj'),
                ('Left GM ACA+MCA _adj', 'left_adj'), ('Left GM PCA _adj', 'left_adj')
            ],
            'ipsilateral-right': [
                ('Right Global ACA+MCA _adj', 'right_adj'), ('Right Global PCA _adj', 'right_adj'),
                ('Right GM ACA+MCA _adj', 'right_adj'), ('Right GM PCA _adj', 'right_adj')
            ],
            'contralateral-left': [
                ('Left Global ACA+MCA _adj', 'right_adj'), ('Left Global PCA _adj', 'right_adj'),
                ('Left GM ACA+MCA _adj', 'right_adj'), ('Left GM PCA _adj', 'right_adj')
            ],
            'contralateral-right': [
                ('Right Global ACA+MCA _adj', 'left_adj'), ('Right Global PCA _adj', 'left_adj'),
                ('Right GM ACA+MCA _adj', 'left_adj'), ('Right GM PCA _adj', 'left_adj')
            ],
            'bilateral': [
                ('Bilateral Global ACA+MCA _adj', 'bilateral_adj'), ('Bilateral Global PCA _adj', 'bilateral_adj'),
                ('Bilateral GM ACA+MCA _adj', 'bilateral_adj'), ('Bilateral GM PCA _adj', 'bilateral_adj')
            ]
        },
        'octa_vs_gm_cbf': {
            'bilateral': [
                ('rel GM CBF _adj', 'RVN6_average_adj'), ('rel GM CBF _adj', 'DVP6_average_adj'),
                ('rel GM CBF _adj', 'SVP6_average_adj')
            ]
        }
    }
    return pairs

def compute_correlations(df, pairs, comparison_type):
    """Compute correlations for each variable separately in ipsilateral/contralateral and bilateral pairs."""
    corr_results = []
    detailed_corr_results = []

    for comp, pair_dict in pairs.items():
        octa_metrics = ['RVN', 'DVP', 'SVP']
        vascular_metrics = ['global_aca+mca', 'global_pca', 'gm_aca+mca', 'gm_pca']

        if comp != 'octa_vs_gm_cbf':
            for lat in ['ipsilateral', 'contralateral']:
                if comp == 'octa_vs_vascular':
                    for v_metric in vascular_metrics:
                        for o_metric in octa_metrics:
                            relevant_pairs = [(x, y) for x, y in pair_dict[lat] if v_metric in x and o_metric in y]
                            combined_data = pd.concat([df[[x, y, 'Source']].rename(columns={x: 'X', y: 'Y'}) 
                                                     for x, y in relevant_pairs])
                            for source in [0, 1]:  # 0: UPenn, 1: UMiami
                                source_name = 'UPenn' if source == 0 else 'UMiami'
                                source_data = combined_data[combined_data['Source'] == source].dropna(subset=['X', 'Y'])
                                if len(source_data) >= 2:
                                    corr, p_value = stats.pearsonr(source_data['X'], source_data['Y'])
                                    x_label = f'{lat}_{v_metric}'
                                    y_label = o_metric
                                    corr_results.append((x_label, y_label, source_name, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
                elif comp == 'octa_vs_obf':
                    for o_metric in octa_metrics:
                        relevant_pairs = [(x, y) for x, y in pair_dict[lat] if o_metric in y]
                        combined_data = pd.concat([df[[x, y, 'Source']].rename(columns={x: 'X', y: 'Y'}) 
                                                 for x, y in relevant_pairs])
                        for source in [0, 1]:
                            source_name = 'UPenn' if source == 0 else 'UMiami'
                            source_data = combined_data[combined_data['Source'] == source].dropna(subset=['X', 'Y'])
                            if len(source_data) >= 2:
                                corr, p_value = stats.pearsonr(source_data['X'], source_data['Y'])
                                x_label = f'{lat}'
                                y_label = o_metric
                                corr_results.append((x_label, y_label, source_name, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
                elif comp == 'vascular_vs_obf':
                    for v_metric in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']:
                        relevant_pairs = [(x, y) for x, y in pair_dict[lat] if v_metric in x]
                        combined_data = pd.concat([df[[x, y, 'Source']].rename(columns={x: 'X', y: 'Y'}) 
                                                 for x, y in relevant_pairs])
                        for source in [0, 1]:
                            source_name = 'UPenn' if source == 0 else 'UMiami'
                            source_data = combined_data[combined_data['Source'] == source].dropna(subset=['X', 'Y'])
                            if len(source_data) >= 2:
                                corr, p_value = stats.pearsonr(source_data['X'], source_data['Y'])
                                x_label = f'{lat}'
                                y_label = v_metric
                                corr_results.append((x_label, y_label, source_name, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))

        for x_col, y_col in pair_dict['bilateral']:
            valid_data = df[[x_col, y_col, 'Source']].dropna(subset=[x_col, y_col])
            for source in [0, 1]:
                source_name = 'UPenn' if source == 0 else 'UMiami'
                source_data = valid_data[valid_data['Source'] == source]
                if len(source_data) >= 2:
                    corr, p_value = stats.pearsonr(source_data[x_col], source_data[y_col])
                    corr_results.append((x_col, y_col, source_name, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
                    detailed_corr_results.append((f'{comp}_bilateral', x_col, y_col, source_name, 
                                                 corr, p_value, 'Yes' if p_value < 0.05 else 'No'))

        for lat in ['ipsilateral', 'contralateral', 'bilateral']:
            for x_col, y_col in pair_dict[lat]:
                valid_data = df[[x_col, y_col, 'Source']].dropna(subset=[x_col, y_col])
                for source in [0, 1]:
                    source_name = 'UPenn' if source == 0 else 'UMiami'
                    source_data = valid_data[valid_data['Source'] == source]
                    if len(source_data) >= 2:
                        corr, p_value = stats.pearsonr(source_data[x_col], source_data[y_col])
                        detailed_corr_results.append((f'{comp}_{lat}', x_col, y_col, source_name, 
                                                     corr, p_value, 'Yes' if p_value < 0.05 else 'No'))

    return corr_results, detailed_corr_results


def compute_correlations_combined_adj(df, pairs, comparison_type):
    """Compute combined correlations for adjusted data, mimicking unadjusted combined sheet."""
    corr_results = []

    for comp, pair_dict in pairs.items():
        octa_metrics = ['RVN', 'DVP', 'SVP']
        vascular_metrics = ['global_aca+mca', 'global_pca', 'gm_aca+mca', 'gm_pca']
        gm_cbf_metrics = ['rel GM CBF']

        for lat in ['ipsilateral-left', 'ipsilateral-right', 'contralateral-left', 'contralateral-right']:
            if comp == 'octa_vs_vascular':
                for v_metric in vascular_metrics:
                    for o_metric in octa_metrics:
                        relevant_pairs = [(x, y) for x, y in pair_dict[lat] if v_metric in x and o_metric in y]
                        combined_data = pd.concat([df[[x, y]].rename(columns={x: 'X', y: 'Y'}) 
                                                 for x, y in relevant_pairs])
                        valid_data = combined_data.dropna(subset=['X', 'Y'])
                        if len(valid_data) >= 2:
                            corr, p_value = stats.pearsonr(valid_data['X'], valid_data['Y'])
                            x_label = f'{lat}_{v_metric}'
                            y_label = o_metric
                            corr_results.append((x_label, y_label, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
            elif comp == 'octa_vs_obf':
                for o_metric in octa_metrics:
                    relevant_pairs = [(x, y) for x, y in pair_dict[lat] if o_metric in y]
                    combined_data = pd.concat([df[[x, y]].rename(columns={x: 'X', y: 'Y'}) 
                                             for x, y in relevant_pairs])
                    valid_data = combined_data.dropna(subset=['X', 'Y'])
                    if len(valid_data) >= 2:
                        corr, p_value = stats.pearsonr(valid_data['X'], valid_data['Y'])
                        x_label = f'{lat}'
                        y_label = o_metric
                        corr_results.append((x_label, y_label, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
            elif comp == 'vascular_vs_obf':
                for v_metric in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']:
                    relevant_pairs = [(x, y) for x, y in pair_dict[lat] if v_metric in x]
                    combined_data = pd.concat([df[[x, y]].rename(columns={x: 'X', y: 'Y'}) 
                                             for x, y in relevant_pairs])
                    valid_data = combined_data.dropna(subset=['X', 'Y'])
                    if len(valid_data) >= 2:
                        corr, p_value = stats.pearsonr(valid_data['X'], valid_data['Y'])
                        x_label = f'{v_metric}'
                        y_label = f'{lat}'
                        corr_results.append((x_label, y_label, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))
            elif comp == 'octa_vs_gm_cbf':
                # Skip for non-bilateral as octa_vs_gm_cbf only has bilateral pairs
                continue

        # Handle bilateral pairs (unchanged)
        for x_col, y_col in pair_dict['bilateral']:
            valid_data = df[[x_col, y_col]].dropna(subset=[x_col, y_col])
            if len(valid_data) >= 2:
                corr, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])
                corr_results.append((x_col, y_col, corr, p_value, 'Yes' if p_value < 0.05 else 'No'))

    return corr_results


def main(apply_site_correction=False):
    """Main function."""
    dmg, octa, cbf, obf, vascular = load_data()
    cbf_processed = preprocess_cbf(cbf)
    octa_processed = preprocess_octa(octa) if octa is not None else None
    obf_processed = preprocess_obf(obf, cbf_processed)
    vascular_processed = preprocess_vascular(vascular, cbf_processed)
    
    df_no_octa = merge_data(dmg, cbf_processed, obf_processed, vascular_processed)
    df_with_octa = merge_data(dmg, cbf_processed, obf_processed, vascular_processed, octa_processed)
    
    if apply_site_correction:
        obf_measures = ['rel GM CBF ', 'GM CBF ', 'bilateral', 'right', 'left', 'rel_bilateral', 'rel_right', 'rel_left']
        vascular_measures = [f'{side} {measure} ' for side in ['Left', 'Right'] for measure in 
                            ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']] + \
                           [f'rel_{side}_{measure.replace(" ", "_").lower()}' for side in ['left', 'right', 'bilateral'] 
                            for measure in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']] + \
                           [f'Bilateral {measure} ' for measure in ['Global ACA+MCA', 'Global PCA', 'GM ACA+MCA', 'GM PCA']]
        octa_measures = ['DVP6_right', 'RVN6_right', 'SVP6_right', 'DVP6_left', 'RVN6_left', 'SVP6_left',
                         'DVP6_average', 'RVN6_average', 'SVP6_average'] if octa_processed is not None else []
        
        df_no_octa = correct_site_effect(df_no_octa, obf_measures, 'Source')
        df_no_octa = correct_site_effect(df_no_octa, vascular_measures, 'Source')
        df_with_octa = correct_site_effect(df_with_octa, obf_measures, 'Source')
        df_with_octa = correct_site_effect(df_with_octa, vascular_measures, 'Source')
        if octa_processed is not None:
            df_with_octa = correct_site_effect(df_with_octa, octa_measures, 'Source')
        # print("Columns in df_with_octa after site correction:", df_with_octa.columns.tolist())  # Debug
        print("\nSite effect correction applied.")
    
    base_path = '/Volumes/Alice_Data/VCID/'
    # result_df_save_path = os.path.join(base_path, 'merged_data_left_right.xlsx')
    # with pd.ExcelWriter(result_df_save_path, engine='openpyxl') as writer:
    #     df_no_octa.to_excel(writer, sheet_name='OBF_CBF_Vascular', index=False)
    #     df_with_octa.to_excel(writer, sheet_name='OBF_CBF_Vascular_OCTA', index=False)
    
    if apply_site_correction:
        pairs = generate_corr_pairs_adj()
        for comp, pair_dict in pairs.items():
            if comp != 'octa_vs_gm_cbf':
                print(f"\nComparison: {comp}")
                for lat in ['ipsilateral-left', 'ipsilateral-right', 'contralateral-left', 'contralateral-right']:
                    num_pairs = len(pair_dict[lat])
                    print(f"  {lat.capitalize()} pairs: {num_pairs}")
        
        corr_results_combined = {}
        corr_results_combined['octa_vs_vascular'] = compute_correlations_combined_adj(
            df_with_octa, {'octa_vs_vascular': pairs['octa_vs_vascular']}, 'octa_vs_vascular')
        corr_results_combined['octa_vs_obf'] = compute_correlations_combined_adj(
            df_with_octa, {'octa_vs_obf': pairs['octa_vs_obf']}, 'octa_vs_obf')
        corr_results_combined['vascular_vs_obf'] = compute_correlations_combined_adj(
            df_no_octa, {'vascular_vs_obf': pairs['vascular_vs_obf']}, 'vascular_vs_obf')
        corr_results_combined['octa_vs_gm_cbf'] = compute_correlations_combined_adj(
            df_with_octa, {'octa_vs_gm_cbf': pairs['octa_vs_gm_cbf']}, 'octa_vs_gm_cbf')
        
        corr_save_path = os.path.join(base_path, 'correlation_results_site_adj_leftright.xlsx')
        with pd.ExcelWriter(corr_save_path, engine='openpyxl') as writer:
            for key, results in corr_results_combined.items():
                pd.DataFrame(results, columns=['X_Measure', 'Y_Measure', 'Correlation', 'P_Value', 'Significant'])\
                    .to_excel(writer, sheet_name=f'{key}_Combined', index=False)
        
        print(f"\nCombined correlation results (site-adjusted) saved to {corr_save_path}")
    else:
        pairs = generate_corr_pairs()
        
        all_results = {}
        all_results['octa_vs_vascular'] = compute_correlations(df_with_octa, {'octa_vs_vascular': pairs['octa_vs_vascular']}, 'octa_vs_vascular')
        all_results['octa_vs_obf'] = compute_correlations(df_with_octa, {'octa_vs_obf': pairs['octa_vs_obf']}, 'octa_vs_obf')
        all_results['vascular_vs_obf'] = compute_correlations(df_no_octa, {'vascular_vs_obf': pairs['vascular_vs_obf']}, 'vascular_vs_obf')
        
        corr_save_path = os.path.join(base_path, 'correlation_results_leftright.xlsx')
        with pd.ExcelWriter(corr_save_path, engine='openpyxl') as writer:
            for key, (corr_results, detailed_results) in all_results.items():
                pd.DataFrame(corr_results, columns=['X_Measure', 'Y_Measure', 'Source', 'Correlation', 'P_Value', 'Significant'])\
                    .to_excel(writer, sheet_name=f'{key}_Combined', index=False)
                pd.DataFrame(detailed_results, columns=['Comparison', 'X_Measure', 'Y_Measure', 'Source', 'Correlation', 'P_Value', 'Significant'])\
                    .to_excel(writer, sheet_name=f'{key}_Detailed', index=False)
        
        print(f"\nCorrelation results saved to {corr_save_path}")
        if not apply_site_correction:
            print("Note: Site effect correction was not applied. Set apply_site_correction=True to enable.")

if __name__ == "__main__":
    main(apply_site_correction=True)