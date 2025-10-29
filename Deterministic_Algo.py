import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.independencies import Independencies
import random
import os

# Record the start time of the entire script
total_start_time = time.time()

# Remove fixed seed to allow variability (optional: comment out if you want some control)
# random.seed(42)  # Commented out for variability

# === Load and Preprocess Data ===
base_path = r"C:\Users\KAVISH\Downloads\Combine_It\All_"
courtship_file = os.path.join(base_path, "Final_Courtship_binned.csv")
male_file = os.path.join(base_path, "Binned_Final_Male.csv")
female_file = os.path.join(base_path, "Female_Final_Binned_1.csv")

# Verify files exist
for file_path in [courtship_file, male_file, female_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

df_courtship = pd.read_csv(courtship_file)
df_male = pd.read_csv(male_file)
df_female = pd.read_csv(female_file)

# Rename columns safely
rename_dict = {
    "1_theta": "theta_1",
    "2_theta": "theta_2",
    "dp_binned": "dp",
    "dc1_binned": "dc1",
    "dc2_binned": "dc2"
}
for df in [df_courtship, df_male, df_female]:
    existing_cols = {col: rename_dict[col] for col in rename_dict if col in df.columns}
    df.rename(columns=existing_cols, inplace=True)

# Define required columns and ensure they exist
columns = ["theta_1", "theta_2", "B1", "B2", "dp", "dc1", "dc2", "Fly_1_sex", "Fly_2_sex"]
available_columns = [col for col in columns if col in df_courtship.columns]
df_courtship = df_courtship[available_columns].dropna()
df_male = df_male[available_columns].dropna()
df_female = df_female[available_columns].dropna()

# === Sampling ===
num_rows = 20000
courtship_rows = min((2 * num_rows) // 4, len(df_courtship))
male_rows = min(num_rows // 4, len(df_male))
female_rows = min(num_rows // 4, len(df_female))

min_ratio = min(courtship_rows // 2, male_rows, female_rows)
courtship_rows = min_ratio * 2
male_rows = female_rows = min_ratio
num_rows = courtship_rows + male_rows + female_rows

df_courtship_sample = df_courtship.sample(n=courtship_rows, random_state=None)  # Remove fixed seed
df_male_sample = df_male.sample(n=male_rows, random_state=None)
df_female_sample = df_female.sample(n=female_rows, random_state=None)

df_selected = pd.concat([df_courtship_sample, df_male_sample, df_female_sample], ignore_index=True)
# Double shuffle
df_selected = df_selected.sample(frac=1, random_state=None).reset_index(drop=True)  # First shuffle
df_selected = df_selected.sample(frac=1, random_state=None).reset_index(drop=True)  # Second shuffle

print(f"Randomly selected {num_rows} rows for training (Courtship: {courtship_rows}, Male: {male_rows}, Female: {female_rows})")

# === Set a single random significance level for both original and mirrored ===
sig_level = random.uniform(0.001, 0.01)  # Stricter significance level, fixed for both runs
print(f"Fixed random significance level chosen for both original and mirrored: {sig_level:.3f}")

# === PC Algorithm on Original (Probe) ===
start_time_original = time.time()
print("\nUsing PC Algorithm for Structure Learning (Original)...")

# Define independencies: dp ⊥ dc1 and Fly_1_sex ⊥ Fly_2_sex
ind = Independencies()
ind.add_assertions(['dp', 'dc1', []])  # dp ⊥ dc1
ind.add_assertions(['Fly_1_sex', 'Fly_2_sex', []])  # Fly_1_sex ⊥ Fly_2_sex

# Drop B2 and dc2 for the original (probe) analysis
data_probe = df_selected.drop(columns=["B2", "dc2"])
try:
    pc = PC(data=data_probe, independencies=ind)
    dag = pc.estimate(ci_test="chi_square", significance_level=sig_level, return_type="dag")
except Exception as e:
    print(f"Error in PC algorithm (original): {e}")
    raise

# Create model and enforce independencies
model = DiscreteBayesianNetwork(dag.edges())
print("\nEdges in original model:", list(model.edges()))

# Enforce dp ⊥ dc1
for u, v in [('dp', 'dc1'), ('dc1', 'dp')]:
    if model.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating dp ⊥ dc1. Removing it...")
        model.remove_edge(u, v)

# Enforce Fly_1_sex ⊥ Fly_2_sex
for u, v in [('Fly_1_sex', 'Fly_2_sex'), ('Fly_2_sex', 'Fly_1_sex')]:
    if model.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating Fly_1_sex ⊥ Fly_2_sex. Removing it...")
        model.remove_edge(u, v)

model.fit(data_probe, estimator=MaximumLikelihoodEstimator)

end_time_original = time.time()
time_original_pc = end_time_original - start_time_original
print(f"\n⏱ Execution Time (Original PC Algorithm): {time_original_pc:.2f} seconds")

# Visualize DAG
plt.figure(figsize=(12, 8))
G = nx.DiGraph(model.edges())
G.add_nodes_from(data_probe.columns)
pos = nx.spring_layout(G, seed=None)  # Remove fixed seed for layout variability
nx.draw(G, pos, with_labels=True, node_color="#f6c1ff", edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
plt.title("Causal DAG with dc1 (PC Algorithm, B2 excluded)")
plt.show()

# === MIRRORED DATA (with dc2) ===
df_mirror = df_selected.rename(columns={
    "theta_1": "theta_2", "theta_2": "theta_1",
    "B1": "B2", "B2": "B1",
    "Fly_1_sex": "Fly_2_sex", "Fly_2_sex": "Fly_1_sex"  # Swap Fly_1_sex and Fly_2_sex
}).copy()

if "dc2" in df_selected.columns:
    df_mirror["dc2"] = df_selected["dc2"].copy()
if "dc1" in df_mirror.columns:
    df_mirror.drop(columns=["dc1"], inplace=True)

# PC Algorithm on Mirrored
start_time_mirrored = time.time()
print("\nUsing PC Algorithm for Structure Learning (Mirrored)...")

# Define independencies: dp ⊥ dc2 and Fly_1_sex ⊥ Fly_2_sex
ind_mirror = Independencies()
ind_mirror.add_assertions(['dp', 'dc2', []])  # dp ⊥ dc2
ind_mirror.add_assertions(['Fly_2_sex', 'Fly_1_sex', []])  # Fly_2_sex ⊥ Fly_1_sex (consistent with original)

# Drop B1 for the mirrored analysis
data_mirror = df_mirror.drop(columns=["B1"])
try:
    pc_mirror = PC(data=data_mirror, independencies=ind_mirror)
    dag_mirror = pc_mirror.estimate(ci_test="chi_square", significance_level=sig_level, return_type="dag")
except Exception as e:
    print(f"Error in PC algorithm (mirrored): {e}")
    raise

# Create model and enforce independencies
model_mirror = DiscreteBayesianNetwork(dag_mirror.edges())
print("\nEdges in mirrored model:", list(model_mirror.edges()))

# Enforce dp ⊥ dc2
for u, v in [('dp', 'dc2'), ('dc2', 'dp')]:
    if model_mirror.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating dp ⊥ dc2. Removing it...")
        model_mirror.remove_edge(u, v)

# Enforce Fly_1_sex ⊥ Fly_2_sex
for u, v in [('Fly_1_sex', 'Fly_2_sex'), ('Fly_2_sex', 'Fly_1_sex')]:
    if model_mirror.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating Fly_1_sex ⊥ Fly_2_sex. Removing it...")
        model_mirror.remove_edge(u, v)

model_mirror.fit(data_mirror, estimator=MaximumLikelihoodEstimator)

end_time_mirrored = time.time()
time_mirrored_pc = end_time_mirrored - start_time_mirrored
print(f"\n⏱ Execution Time (Mirrored PC Algorithm): {time_mirrored_pc:.2f} seconds")

# Visualize Mirrored DAG
plt.figure(figsize=(12, 8))
G_mirror = nx.DiGraph(model_mirror.edges())
G_mirror.add_nodes_from(data_mirror.columns)
pos_mirror = nx.spring_layout(G_mirror, seed=None)  # Remove fixed seed for layout variability
nx.draw(G_mirror, pos_mirror, with_labels=True, node_color="#c1f6e2", edge_color='black', node_size=3000, font_size=10, font_weight='bold')
plt.title("Mirrored Causal DAG with dc2 (PC Algorithm, B1 excluded)")
plt.show()

# === COMBINE NETWORKS ===
combined_edges = list(set(model.edges()).union(set(model_mirror.edges())))
G_combined = nx.DiGraph()
G_combined.add_edges_from(combined_edges)

# Use the columns from the probe and mirror datasets directly
combined_nodes = set(data_probe.columns).union(set(data_mirror.columns))
G_combined.add_nodes_from(combined_nodes)
pos_combined = nx.spring_layout(G_combined, seed=None)  # Remove fixed seed for layout variability

edge_colors = []
for u, v in G_combined.edges():
    if (u, v) in model.edges() and (u, v) in model_mirror.edges():
        edge_colors.append('mediumpurple')
    elif (u, v) in model.edges():
        edge_colors.append('skyblue')
    elif (u, v) in model_mirror.edges():
        edge_colors.append('salmon')
    else:
        edge_colors.append('gray')

plt.figure(figsize=(16, 10))
nx.draw(
    G_combined, pos_combined, with_labels=True,
    node_color="#e0e0e0", edge_color=edge_colors,
    node_size=3200, font_size=10, font_weight='bold'
)
plt.title("Final Combined Causal DAG (Original with dc1 + Mirrored with dc2)", fontsize=14)
plt.show()

# === Final Summary ===
total_end_time = time.time()
total_execution_time = total_end_time - total_start_time
total_pc_time = time_original_pc + time_mirrored_pc

print("\n✅ Final Combined DAG Summary:")
print(f"🔹 Total unique edges: {len(combined_edges)}")
print("🔹 Edge color legend:")
print("   - 🔵 skyblue: Original only (dc1, B1 included)")
print("   - 🔴 salmon: Mirrored only (dc2, B2 included)")
print("   - 🟣 mediumpurple: Shared in both")
print(f"⏱ Execution Time (Original PC Algorithm): {time_original_pc:.2f} seconds")
print(f"⏱ Execution Time (Mirrored PC Algorithm): {time_mirrored_pc:.2f} seconds")
print(f"⏱ Total Execution Time (All PC Algorithms): {total_pc_time:.2f} seconds")
print(f"⏱ Total Execution Time (Entire Script): {total_execution_time:.2f} seconds")