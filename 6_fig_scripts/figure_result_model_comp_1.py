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

df = pd.read_csv('../3_Results/_comparison/fossil_model_comparison_report_v2.csv')
metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'auc', 'top3']
df_melted = df.melt(id_vars='model', value_vars=metrics, var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 7))
# Use colorblind palette
palette = sns.color_palette('colorblind', n_colors=len(metrics))
ax = sns.barplot(data=df_melted, x='model', y='Score', hue='Metric', palette=palette, edgecolor='black')

# Remove top and right spines
sns.despine(ax=ax)

plt.ylim(0.8, 1.0)
plt.title('Model Comparison Across Key Metrics', fontsize=13, weight='bold', pad=10)
plt.ylabel('Score', fontsize=12, weight='bold')
plt.xlabel('Model', fontsize=12, weight='bold')

# Place legend outside the plot
plt.legend(title='Metric', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout(pad=1.5)
plt.savefig('./plots/model_comparison_metrics.svg')
plt.show()

# print the data frame for latex table
print(df[['model','accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'top3', 'auc']].to_latex(index=False, float_format="%.4f"))