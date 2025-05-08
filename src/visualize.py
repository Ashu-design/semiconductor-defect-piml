# -----------------------------------
# Imports for Visualization
# -----------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.style.use('default')

# -----------------------------------
# 1. Distribution Plots
# -----------------------------------

# Plot feature distributions
plt.figure(figsize=(18, 12))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(X[feature], kde=True)
    plt.title(f'Distribution: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# -----------------------------------
# Improved Correlation Heatmap
# -----------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation
corr_matrix = X.corr()

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='RdBu_r',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.75},
    annot_kws={"size": 10}
)
plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

# -----------------------------------
# 3. Target vs Feature Scatterplots
# -----------------------------------
plt.figure(figsize=(18, 12))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=X[feature], y=y)
    plt.title(f'{feature} vs HSE Defect Level')
    plt.xlabel(feature)
    plt.ylabel('HSE Defect Level [eV]')

plt.tight_layout()
plt.show()

# -----------------------------------
# 4. Pairplot (optional but heavy if >1000 points)
# -----------------------------------
# You can downsample to ~500 points for faster rendering if needed
sample_df = df.sample(n=min(500, df.shape[0]), random_state=42)

sns.pairplot(sample_df[selected_features + [target]], diag_kind="kde", corner=True)
plt.suptitle('Pairplot of Features and Target (Sampled)', y=1.02)
plt.show()


# -----------------------------------
# PCA 3D Visualization
# -----------------------------------
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
import matplotlib.pyplot as plt

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Variance explained
print("\nExplained Variance Ratios by 3 Components:")
print(pca.explained_variance_ratio_)
print(f"Total Variance Captured: {np.sum(pca.explained_variance_ratio_):.2f}")

# Create a DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['hse_defect_level'] = y.values  # Add target for coloring

# 3D Scatter Plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Scatter
sc = ax.scatter(
    pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
    c=pca_df['hse_defect_level'],
    cmap='viridis',  # Or 'coolwarm'
    alpha=0.7
)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Features (Color = HSE Defect Level)')

# Color bar
cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('HSE Defect Level [eV]')

plt.tight_layout()
plt.show()


# -----------------------------------
# Feature Importance using Mutual Information
# -----------------------------------
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculate Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Display
print("\n Mutual Information Feature Importance Scores:")
print(mi_scores)

# Plot
plt.figure(figsize=(10,6))
mi_scores.plot(kind='barh')
plt.title('Feature Importance based on Mutual Information')
plt.xlabel('Mutual Information Score')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# -----------------------------------
# Force Plot using matplotlib and save as PNG
# -----------------------------------
import shap
import matplotlib.pyplot as plt

# Pick sample
instance_idx = 0

# Matplotlib-based force plot
plt.figure(figsize=(16, 3))  # Wider figure to avoid squishing
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[instance_idx],
    features=X_sample[instance_idx],
    feature_names=selected_features,
    matplotlib=True  # Important!
)
plt.tight_layout()

# Save as PNG
plt.savefig('/content/force_plot_instance.png', dpi=300, bbox_inches='tight')
print("✅ Force Plot saved as /content/force_plot_instance.png")
plt.show()


# -----------------------------------
# Batch Force Plot Saving (Top 5 Features Only)
# -----------------------------------
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

# Parameters
num_samples = 10   # How many samples to save
num_top_features = 5  # Top 5 features shown in each Force Plot

# Create folder
output_folder = '/content/force_plots_batch'
os.makedirs(output_folder, exist_ok=True)

# Function to save one force plot
def save_force_plot(instance_idx):
    feature_impact = np.abs(shap_values[instance_idx])
    top_feature_idx = np.argsort(feature_impact)[-num_top_features:]
    top_features = [selected_features[i] for i in top_feature_idx]

    filtered_features = X_sample[instance_idx][top_feature_idx]
    filtered_shap_values = shap_values[instance_idx][top_feature_idx]

    plt.figure(figsize=(10,3))
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=filtered_shap_values,
        features=filtered_features,
        feature_names=top_features,
        matplotlib=True
    )
    plt.tight_layout()
    save_path = os.path.join(output_folder, f"force_plot_sample_{instance_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------
# Save Force Plots for First N Samples
# -----------------------------------
for idx in range(num_samples):
    save_force_plot(idx)

print(f"\n✅ Force Plots saved for {num_samples} samples inside: {output_folder}")


# -----------------------------------
# Force Plot HTML Save (Best Quality)
# -----------------------------------
import shap
from IPython.display import display, HTML

instance_idx = 0

# Full Force Plot
force_plot = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[instance_idx],
    features=X_sample[instance_idx],
    feature_names=selected_features
)

# Save as HTML
html_content = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

with open("/content/force_plot_full.html", "w") as f:
    f.write(html_content)

print("Force Plot saved as /content/force_plot_full.html (Open in browser and take PNG screenshot)")


# -----------------------------------
# Top 3 Features Dependence Plots
# -----------------------------------

# Find Top 3 Important Features
import numpy as np
feature_importance = np.abs(shap_values).mean(axis=0)
top3_idx = np.argsort(feature_importance)[-3:][::-1]
top3_features = [selected_features[i] for i in top3_idx]

print("\nTop 3 Important Features by SHAP:")
print(top3_features)

# Plot Dependence Plots
for feature_name in top3_features:
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample,
        feature_names=selected_features
    )


shap.dependence_plot(
    feature_name,
    shap_values,
    X_sample,
    feature_names=selected_features,
    interaction_index='auto'  # Let SHAP auto-detect interaction
)
