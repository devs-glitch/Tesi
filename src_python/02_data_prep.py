'''Read Metadata, filter what we need, organize files for Pytorch'''

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import sklearn
from sklearn.preprocessing import StandardScaler
import umap


H1_O1 = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\H1_O1.csv")
H1_O2 = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\H1_O2.csv")
H1_O3a = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\H1_O3a.csv")
H1_O3b = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\H1_O3b.csv")
L1_O1 = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\L1_O1.csv")
L1_O2 = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\L1_O2.csv")
L1_O3a = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\L1_O3a.csv")
L1_O3b = pd.read_csv(r"C:\Users\devam\OneDrive\Tesi\data\raw\L1_O3b.csv")

print(H1_O1)
print(H1_O2)
print(H1_O3a)
print(H1_O3b)
print(L1_O1)
print(L1_O2)
print(L1_O3a)
print(L1_O3b)

# Unify all datasets in df

base_path = r"C:\Users\devam\OneDrive\Tesi\data\raw"
file_names = ["H1_O1", "H1_O2", "H1_O3a", "H1_O3b", "L1_O1", "L1_O2", "L1_O3a", "L1_O3b"]

data = {}
all_data_list = []

for name in file_names:
    file_path = os.path.join(base_path, f"{name}.csv")
    df = pd.read_csv(file_path)
    
    # Add a column to keep track of which file comes from each string
    df['detector_run'] = name 
    
    data[name] = df
    all_data_list.append(df)

df = pd.concat(all_data_list, ignore_index=True)

df.head()
print(f"Dataset shape: {df.shape}")
df.info()

# Check also unique values and null values
df.nunique()
df.isnull().sum()

'''
We want to perform a little analysis on the labels counts, since we want to keep only the 3 to 4 most frequent ones
'''

label_column = 'ml_label'

label_counts = df[label_column].value_counts()

print("Class Distribution")
print(label_counts)

plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('# campioni per label')
plt.xlabel('label')
plt.ylabel('# campioni')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

'''
FEATURE SELECTION
'''
# Generate box plots for numerical features
# --> how a single feature can separate classes

numeric_columns = ['event_time', 'peak_time', 'peak_time_ns', 'start_time', 'start_time_ns', 'duration', 'peak_frequency', 'central_freq', 'amplitude', 'snr', 'bandwidth', 'q_value']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col, fontsize=12, pad=20)
    axes[i].set_ylabel("")
    axes[i].tick_params(axis='y', labelsize=9)

    axes[i].yaxis.get_offset_text().set_fontsize(8)
    axes[i].yaxis.get_offset_text().set_y(1.02)

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.5)
plt.savefig("Feature_Box_Plots.png", dpi=150, bbox_inches="tight")
plt.show()


# Compute correlation matrix (numeric columns)
# --> whether two+ feature mean the same thing

correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correleation Matrix")
plt.tight_layout()
plt.show()

'''
We now select only our most distiguisable classes, derived from the analysis above.
we will select: Scattered_Light, Blip, Extremely_Loud, and Violin_Mode.
'''
selected_classes = ["Scattered_Light", "Blip", "Extremely_Loud", "Violin_Mode"]

# We want to visualize the relationship between different feature wrt the class label
selected_features = ['duration', 'peak_frequency', 'snr', 'bandwidth', 'ml_label']

df_pairplot = df[df["ml_label"].isin(selected_classes)][selected_features].copy()
df_pairplot["snr"] = np.log1p(df_pairplot["snr"])
df_pairplot["duration"] = np.log1p(df_pairplot["duration"])

g = sns.pairplot(
    data=df_pairplot,
    hue="ml_label",
    palette="Set1",
    corner=True,
    height=3.5,
    plot_kws={'alpha': 0.5, 's': 15, 'linewidth': 0},
    diag_kws={'fill': True, 'alpha': 0.3}
)

for ax in g.axes.flatten():
    if ax is None:
        continue
    if ax.get_xlabel() in ("snr", "duration"):
        ax.set_xlabel(f"log({ax.get_xlabel()} + 1)")
    if ax.get_ylabel() in ("snr", "duration"):
        ax.set_ylabel(f"log({ax.get_ylabel()} + 1)")

sns.move_legend(g, "center right", bbox_to_anchor=(1.02, 0.5), title="Class labels", fontsize=10)
plt.suptitle("Multivariate analysis: duration, peak_frequency, snr, bandwidth vs class labels",
             y=1.02, fontsize=16)
plt.savefig("Figure_4_Feature_Pairplots_Selected_Classes.png", dpi=150, bbox_inches="tight")
plt.show()

'''
We will pass our selected features to have our final df
'''

columns_to_keep = [
    'gravityspy_id',                                            # measurements ID
    'duration', 'peak_frequency', 'snr', 'bandwidth',           # numeric features
    'ml_label',                                                 # labels
    'url1',                                                     # spectrograms
    'Scattered_Light', 'Blip', 'Extremely_Loud', 'Violin_Mode'  # confidence scores
]

mask = df["ml_label"].isin(selected_classes)

selected_df = df.loc[mask, columns_to_keep].reset_index(drop=True)

print(f"Dimensions selected dataset: {selected_df.shape}")
print("\nClass counts:")
print(selected_df["ml_label"].value_counts())
print(selected_df.head())

'''
Handle url=? values
'''

mask1 = selected_df["url1"] != "?"
selected_df = selected_df.loc[mask1].reset_index(drop=True)

'''
Handle class imbalance
'''
min_samples = selected_df['ml_label'].value_counts().min()
selected_df = selected_df.groupby('ml_label').sample(n=min_samples, random_state=42).reset_index(drop=True)


'''
    Clustering
        1. Standardization: apply `StandardScaler` on numeric data
        2. UMAP projection: apply a dimensionality reduction algorithm on tabular data
        3. Plot 2d results
'''

df_sample = selected_df.groupby('ml_label').sample(n=500, random_state=42)
X = df_sample[['duration', 'peak_frequency', 'snr', 'bandwidth']]
y = df_sample['ml_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
embedding = reducer.fit_transform(X_scaled)

df_umap = pd.DataFrame({'UMAP_1': embedding[:, 0], 'UMAP_2': embedding[:, 1], 'Class': y.values})

# plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_umap, 
    x='UMAP_1', 
    y='UMAP_2', 
    hue='Class',       
    palette='Set1',    
    s=20,              
    alpha=0.8          
)

plt.title('UMAP projection on in the 4 selected_classes', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

'''Save new df'''
selected_df.to_csv("data/processed/selected_dataset.csv", index=False)