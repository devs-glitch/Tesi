'''Read Metadata, filter what we need, organize files for Pytorch'''

import pandas as pd
import os
import matplotlib.pyplot as plt



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

'''
We want to perform a little analysis on the labels counts, since we want to keep only the 3 to 4 most frequent ones
'''
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

# Unify all datasets in df
df = pd.concat(all_data_list, ignore_index=True)

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