import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Nature-like style
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

# Load the CSV file as a pivot table (index column is class names)
#df = pd.read_csv('../3_Results/_comparison/fossil_model_per_class_metrics_v2.csv', index_col=0)

# Select only metric columns (adjust as needed)
# metrics = ['precision', 'recall', 'f1']  # Update if your columns have different names
# df_metrics = df[metrics]

# plt.figure(figsize=(10, max(6, len(df_metrics) * 0.5)))
# sns.heatmap(df_metrics, annot=True, cmap='YlGnBu', fmt=".2f")
# plt.title('Per-Class Metrics Heatmap')
# plt.ylabel('Class')
# plt.xlabel('Metric')
# plt.tight_layout()
# plt.savefig('./plots/per_class_metrics_heatmap.svg')

# Read the CSV, skipping the first row, using the second row as header
df = pd.read_csv('../3_Results/_comparison/fossil_model_per_class_metrics_v2.csv', header=[0,1], index_col=0)

# Drop 'support' columns (optional, if you only want precision/recall/f1)
# df = df.loc[:, ~df.columns.str.contains('support')]


# Prepare long-form DataFrames for each metric
metrics = ['precision', 'recall', 'f1']
long_dfs = {}
for metric in metrics:
	metric_df = df.loc[:, df.columns.get_level_values(1) == metric]
	long_df = metric_df.reset_index().melt(id_vars='index', var_name='model', value_name=metric)
	long_df.rename(columns={'index': 'class'}, inplace=True)
	long_dfs[metric] = long_df

# Set up subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
palette = sns.color_palette('colorblind', n_colors=long_dfs['f1']['class'].nunique())

# Get unique classes and assign half to 'o' (circle), half to '*' (star)
unique_classes = sorted(long_dfs['f1']['class'].unique())
# Assign a unique marker to each class for better distinction
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
# Repeat markers if there are more classes than markers
class_markers = {cls: markers[i % len(markers)] for i, cls in enumerate(unique_classes)}

for ax, metric in zip(axes, metrics):
	for cls in unique_classes:
		data = long_dfs[metric][long_dfs[metric]['class'] == cls]
		sns.scatterplot(
			data=data,
			x='model', y=metric, label=cls,
			color=palette[unique_classes.index(cls)],
			marker=class_markers[cls],
			s=80, ax=ax
		)
	# ax.set_title(f'{metric.capitalize()} per Model and Species')
	ax.set_title(f'{metric.capitalize()}')

	ax.set_xlabel('Model')
	if ax == axes[0]:
		ax.set_ylabel('Value')
	else:
		ax.set_ylabel('')
	ax.set_ylim(0, 1.05)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
	ax.grid(True, linestyle='--', alpha=0.7)
	# Only show legend on the last subplot
	if ax != axes[-1]:
		ax.get_legend().remove()
	else:
		ax.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('./plots/model_per_species_metrics.svg')

plt.show()

