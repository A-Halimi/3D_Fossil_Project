import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme(context='paper', style='white')
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.labelweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.dpi': 300,
})

# Helper to parse sklearn classification_report txt files
def parse_classification_report(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Find the start of the table
    # print (lines)
    start = 0
    for i, line in enumerate(lines):
        if re.match(r'\s*precision\s+recall\s+f1-score', line):
            start = i + 1
            break
    # Read all lines after the header, skipping dashes and empty lines
    data = []
    for line in lines[start:]:
        # print(line)
        if re.match(r'\\s*-+\\s*$', line):
            continue
        if line.strip() == '':
            continue
        parts = re.split(r'\\s+', line.strip())
        # print (parts[0].split())
        # print (len(parts), parts, parts[0].split() )
        parts = parts[0].split()
        # Handle accuracy row (has only 2 columns: 'accuracy' and value)
        if 'accuracy' in parts[0]:
            data.append([parts[0], None, None, parts[1], parts[2] if len(parts) > 2 else None])
        # Handle normal and avg rows (should have at least 4 columns)
        elif len(parts) >= 4:
            print (parts)
            if 'macro' in parts[0] or 'weighted' in parts[0]:
                # print("Found macro row:", parts)
                support = parts[-1] if len(parts) > 4 else None
                data.append([parts[0], parts[2], parts[3], parts[4], support])
            else:
                # If support is present, keep it; else, fill with None
                support = parts[4] if len(parts) > 4 else None
                data.append([parts[0], parts[1], parts[2], parts[3], support])
    df = pd.DataFrame(data, columns=['class', 'precision', 'recall', 'f1', 'support'])
    # print (df.head())
    for col in ['precision', 'recall', 'f1', 'support']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# File paths
files = [
    '../3_Results/Ensemble_Top_k/classification_report_ensemble_top-2.txt',
    '../3_Results/Ensemble_Top_k/classification_report_ensemble_top-3.txt',
    '../3_Results/Ensemble_Top_k/classification_report_ensemble_top-4.txt',
    '../3_Results/fossil_classifier_final/Classification_report_Patched Ensemble.txt'
]
labels = ['Top-2', 'Top-3', 'Top-4', 'Patched Ensemble']

# Parse all reports and add a column for the ensemble type
dfs = []
for file, label in zip(files, labels):
    df = parse_classification_report(file)
    # print (df.head())
    df['ensemble'] = label
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
# print(all_df)
# Prepare long-form DataFrames for each metric
metrics = ['precision', 'recall', 'f1']
long_dfs = {}
for metric in metrics:
    long_df = all_df[['class', 'ensemble', metric]].copy()
    long_df = long_df.rename(columns={metric: 'value'})
    long_dfs[metric] = long_df

# Set up subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
palette = sns.color_palette('colorblind', n_colors=all_df['class'].nunique())


for ax, metric in zip(axes, metrics):
    # Always set hue to 'class' and palette
    plot = sns.scatterplot(
        data=long_dfs[metric],
        x='ensemble', 
        y='value', 
        hue='class', 
        palette=palette, 
        s=80, edgecolor='k', ax=ax
    )
    # ax.set_title(f'{metric.capitalize()} per Ensemble and Class')
    ax.set_title(f'{metric.capitalize()}')

    ax.set_xlabel('Ensemble')
    if ax == axes[0]:
        ax.set_ylabel(metric.capitalize())
    else:
        ax.set_ylabel('')
    ax.set_ylim(0, 1.05)
    # Only show legend on the last subplot
    legend = ax.get_legend()
    if ax != axes[-1]:
        if legend is not None:
            legend.remove()
    else:
        if legend is not None:
            legend.set_title('Class')
            legend.set_bbox_to_anchor((1.05, 1))
            # legend.set_loc('upper left')


plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Print LaTeX table for the combined DataFrame (all_df)
print("\nLaTeX table for all ensemble results (precision, recall, f1):\n")
# Pivot to have ensemble as columns, index as class, and multiindex columns for metrics
latex_df = all_df.pivot_table(index='class', columns='ensemble', values=['precision', 'recall', 'f1'])
latex_str = latex_df.to_latex(float_format="{:.3f}".format, multirow=True, na_rep="--")
print(latex_str)

# get the precision, recall, f1 for only Orbitoides and Baculogypsina
print("\nLaTeX table for Orbitoides and Baculogypsina only:\n")
latex_df = all_df[all_df['class'].isin(['Orbitoides', 'Baculogypsina'])].pivot_table(index='class', columns='ensemble', values=['precision', 'recall', 'f1'])
latex_str = latex_df.to_latex(float_format="{:.3f}".format, multirow=True, na_rep="--")
print(latex_str)

#using the dataframe lets plot the precision, recall, f1 for only Orbitoides and Baculogypsina
# suggest the best plor for vizualization that is not a bar plot as there are only two classes
# a scatter plot with lines connecting the points for each metric across ensembles
# or a line plot with markers for each class across ensembles for each metric
# or a grouped bar plot with bars for each class for each ensemble for each metric
# or a heatmap with ensembles on x-axis, classes on y-axis, and color intensity representing the metric value
# or a radar chart with axes for each ensemble and lines for each class
# or a dot plot with ensembles on x-axis, classes on y-axis, and dot size representing the metric value

fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
palette = sns.color_palette('colorblind', n_colors=2)  # Only two classes
classes_of_interest = ['Orbitoides', 'Baculogypsina']
for ax, metric in zip(axes, metrics):
    for cls in classes_of_interest:
        data = long_dfs[metric][(long_dfs[metric]['class'] == cls) & (long_dfs[metric]['class'].isin(classes_of_interest))]
        sns.lineplot(
            data=data,
            x='ensemble', y='value', label=cls,
            marker='o',
            color=palette[classes_of_interest.index(cls)],
            ax=ax
        )
    ax.set_title(f'{metric.capitalize()}')
    ax.set_xlabel('Ensemble')
    if ax == axes[0]:
        # ax.set_ylabel(metric.capitalize())
        ax.set_ylabel('')

    else:
        ax.set_ylabel('')
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)
    # Only show legend on the last subplot
    if ax != axes[-1]:
        ax.get_legend().remove()
    else:
        # ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1])
# save the plot 
plt.savefig('./plots/ensemble_topk_scatter_orbitoides_baculogypsina.svg')
plt.show()


