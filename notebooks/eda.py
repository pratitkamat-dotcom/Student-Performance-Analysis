import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style for clean aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Path definitions
DATASET_PATH = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset\student_data.csv"
IMAGES_DIR = r"C:\student -performance-analyzer\Student-Performance-Analysis\static\images"

os.makedirs(IMAGES_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATASET_PATH)

# List of numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 1. Distribution of the Target Variable (G3)
plt.figure(figsize=(10, 6))
sns.histplot(df['G3'], kde=True, bins=21, color='#4A90E2', edgecolor='black')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3) - Scale 0 to 20')
plt.ylabel('Count of Students')
plt.axvline(df['G3'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['G3'].mean():.2f}")
plt.axvline(df['G3'].median(), color='green', linestyle='-', linewidth=2, label=f"Median: {df['G3'].median():.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'distribution_target_g3.png'), dpi=150)
plt.close()

# 2. Histograms of Key Numerical Variables
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
key_nums = ['age', 'absences', 'studytime', 'traveltime', 'failures', 'health']
colors = ['#FF6B6B', '#4ECDC4', '#45B6FE', '#FFD166', '#8338EC', '#3A86C8']

for ax, col, color in zip(axes.flatten(), key_nums, colors):
    sns.histplot(df[col], kde=False, color=color, ax=ax, discrete=True if col != 'absences' else False)
    ax.set_title(f'Distribution of {col.capitalize()}')
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel('Count')

plt.suptitle('Histograms of Key Numerical Features', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'histograms_numerical.png'), dpi=150)
plt.close()

# 3. Count Plots for Key Categorical Columns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
key_cats = ['school', 'sex', 'address', 'famsize', 'internet', 'romantic']

for ax, col in zip(axes.flatten(), key_cats):
    sns.countplot(x=col, data=df, ax=ax, palette='Set2')
    ax.set_title(f'Count of Students by {col.capitalize()}')
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel('Count')

plt.suptitle('Count Plots of Key Categorical Features', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'countplots_categorical.png'), dpi=150)
plt.close()

# 4. Box Plots (G3 vs Key Categorical/Ordinal variables)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
boxplot_features = ['studytime', 'failures', 'schoolsup', 'address']

for ax, col in zip(axes.flatten(), boxplot_features):
    sns.boxplot(x=col, y='G3', data=df, ax=ax, palette='Set3')
    ax.set_title(f'Final Grade (G3) by {col.capitalize()}')
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel('Final Grade (G3)')

plt.suptitle('Final Grade (G3) Distributions Across Selected Features', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'boxplots_g3_by_features.png'), dpi=150)
plt.close()

# 5. Correlation Heatmap (All Numeric columns)
plt.figure(figsize=(14, 12))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()

# 6. Pair Plot (Key Numerical / Grade Indicators)
pair_plot_features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
g = sns.pairplot(df[pair_plot_features], diag_kind='kde', plot_kws={'alpha': 0.6})
g.fig.suptitle('Pair Plot of Primary Grade Predicting Features', y=1.02)
plt.tight_layout()
g.savefig(os.path.join(IMAGES_DIR, 'pair_plot.png'), dpi=150)
plt.close()

print("All charts generated and saved successfully!")
