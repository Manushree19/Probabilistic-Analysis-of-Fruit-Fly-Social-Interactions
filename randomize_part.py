import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from itertools import combinations
from scipy.stats import chi2_contingency

# === Custom Randomized PC Algorithm ===
class RandomizedPC:
    def __init__(self, data, alpha):
        self.data = data
        self.variables = list(data.columns)
        self.alpha = alpha
        self.G = nx.Graph()  # Start with undirected graph
        self.G.add_nodes_from(self.variables)

    def _independence_test(self, x, y, cond_set, data):
        contingency = pd.crosstab(data[x], [data[y]] + [data[c] for c in cond_set])
        _, p_value, _, _ = chi2_contingency(contingency)
        return p_value

    def estimate(self):
        # Initialize fully connected graph
        for i, var1 in enumerate(self.variables):
            for var2 in self.variables[i+1:]:
                self.G.add_edge(var1, var2)

        # Shuffle variables for randomness
        random.shuffle(self.variables)
        print(f"🔀 Using significance level: {self.alpha}")

        # Learn skeleton with shuffled edge testing
        self.G = self._learn_skeleton()

        # Shuffle edges before orientation for additional randomness
        edges = list(self.G.edges())
        random.shuffle(edges)
        self.G = nx.Graph()
        self.G.add_nodes_from(self.variables)
        self.G.add_edges_from(edges)

        directed_G = self._orient_edges()
        return directed_G

    def _learn_skeleton(self):
        G = self.G.copy()
        n_vars = len(self.variables)
        for size in range(n_vars - 1):
            edge_list = list(G.edges())
            random.shuffle(edge_list)  # Shuffle edges for each iteration
            for x, y in edge_list:
                if not G.has_edge(x, y):
                    continue
                neighbors = set(G.neighbors(x)).union(G.neighbors(y)) - {x, y}
                cond_sets = list(combinations(neighbors, size))
                random.shuffle(cond_sets)
                for cond_set in cond_sets[:min(len(cond_sets), 10)]:
                    p_value = self._independence_test(x, y, cond_set, self.data)
                    if p_value > self.alpha:
                        G.remove_edge(x, y)
                        break
        return G

    def _orient_edges(self):
        directed_G = nx.DiGraph()
        directed_G.add_nodes_from(self.variables)

        # Shuffle edges again for orientation
        edges = list(self.G.edges())
        random.shuffle(edges)

        for u, v in edges:
            directed_G.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(directed_G):
                directed_G.remove_edge(u, v)
                directed_G.add_edge(v, u)
                if not nx.is_directed_acyclic_graph(directed_G):
                    directed_G.remove_edge(v, u)

        # V-structure detection with shuffled nodes
        random.shuffle(self.variables)
        for z in self.variables:
            neighbors = list(self.G.neighbors(z))
            random.shuffle(neighbors)
            for x, y in combinations(neighbors, 2):
                if not self.G.has_edge(x, y):
                    if directed_G.has_edge(z, x):
                        directed_G.remove_edge(z, x)
                    if directed_G.has_edge(z, y):
                        directed_G.remove_edge(z, y)
                    directed_G.add_edge(x, z)
                    directed_G.add_edge(y, z)
                    if not nx.is_directed_acyclic_graph(directed_G):
                        directed_G.remove_edge(x, z)
                        directed_G.remove_edge(y, z)

        return directed_G

# === Load and Preprocess Data ===
base_path = r"C:\Users\KAVISH\Downloads\Combine_It\All_"
courtship_file = base_path + r"\Final_Courtship_binned.csv"
male_file = base_path + r"\Binned_Final_Male.csv"
female_file = base_path + r"\Female_Final_Binned_1.csv"

df_courtship = pd.read_csv(courtship_file)
df_male = pd.read_csv(male_file)
df_female = pd.read_csv(female_file)

for df in [df_courtship, df_male, df_female]:
    df.rename(columns={
        "1_theta": "theta_1",
        "2_theta": "theta_2",
        "dp_binned": "dp",
        "dc1_binned": "dc1",
        "dc2_binned": "dc2"
    }, inplace=True)

columns = ["theta_1", "theta_2", "B1", "B2", "dp", "dc1", "dc2", "Fly_1_sex", "Fly_2_sex"]
df_courtship = df_courtship[columns].dropna()
df_male = df_male[columns].dropna()
df_female = df_female[columns].dropna()

# === Sampling with Double Shuffle ===
num_rows = 20000
courtship_rows = min((2 * num_rows) // 4, len(df_courtship))
male_rows = min(num_rows // 4, len(df_male))
female_rows = min(num_rows // 4, len(df_female))

min_ratio = min(courtship_rows // 2, male_rows, female_rows)
courtship_rows = min_ratio * 2
male_rows = female_rows = min_ratio
num_rows = courtship_rows + male_rows + female_rows

df_courtship_sample = df_courtship.sample(frac=1).reset_index(drop=True).sample(frac=1).reset_index(drop=True).head(courtship_rows)
df_male_sample = df_male.sample(frac=1).reset_index(drop=True).sample(frac=1).reset_index(drop=True).head(male_rows)
df_female_sample = df_female.sample(frac=1).reset_index(drop=True).sample(frac=1).reset_index(drop=True).head(female_rows)

df_selected = pd.concat([df_courtship_sample, df_male_sample, df_female_sample], ignore_index=True)
print(f"Randomly selected {num_rows} rows for training (Courtship: {courtship_rows}, Male: {male_rows}, Female: {female_rows})")

df_selected["dp"] = df_selected["dp"].astype("category")
df_selected["dc1"] = df_selected["dc1"].astype("category")
df_selected["dc2"] = df_selected["dc2"].astype("category")

# === Generate a single random alpha ===
alpha_range = (0.01, 0.1)
fixed_alpha = round(random.uniform(*alpha_range), 3)
print(f"🔀 Fixed significance level for both original and mirrored runs: {fixed_alpha}")

# === Store removed edges ===
removed_edges = {"original": [], "mirrored": []}

# === Randomized PC Algorithm on Original ===
start_time_original = time.time()
print("\nUsing Randomized PC Algorithm for Structure Learning (Original)...")

rpc = RandomizedPC(data=df_selected.drop(columns=["dc2", "B2"]), alpha=fixed_alpha)
dag = rpc.estimate()
model = DiscreteBayesianNetwork(dag.edges())

print("\nEdges before enforcing prior independencies:", list(model.edges()))

# Enforce prior independence: dp ⊥ dc1
for u, v in [('dp', 'dc1'), ('dc1', 'dp')]:
    if model.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating dp ⊥ dc1. Removing it...")
        removed_edges["original"].append((u, v))
        model.remove_edge(u, v)

# Enforce prior independence: Fly_1_sex ⊥ Fly_2_sex
for u, v in [('Fly_1_sex', 'Fly_2_sex'), ('Fly_2_sex', 'Fly_1_sex')]:
    if model.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} detected, violating Fly_1_sex ⊥ Fly_2_sex. Removing it...")
        removed_edges["original"].append((u, v))
        model.remove_edge(u, v)

print("Edges after enforcing prior independencies:", list(model.edges()))

model.fit(df_selected.drop(columns=["dc2", "B2"]), estimator=MaximumLikelihoodEstimator)

end_time_original = time.time()
print(f"\n⏱ Execution Time (Original Randomized PC Algorithm): {end_time_original - start_time_original:.2f} seconds")

# Visualize Original DAG
plt.figure(figsize=(12, 8))
G = nx.DiGraph(model.edges())
G.add_nodes_from(df_selected.drop(columns=["dc2", "B2"]).columns)
pos = nx.spring_layout(G, seed=42, k=1.5)
nx.draw(G, pos, with_labels=True, node_color='lightsteelblue', edge_color='gray', node_size=3000, font_size=12, font_weight='bold')
plt.title("Causal DAG using Randomized PC Algorithm (Original with dc1, without B2)", fontsize=14)
plt.tight_layout()
plt.show()

# === MIRRORED DATA (with dc2) ===
print("\nUsing PC Algorithm for Structure Learning (Mirrored)...")
df_mirror = df_selected.rename(columns={
    "theta_1": "theta_2", "theta_2": "theta_1",
    "B1": "B2", "B2": "B1",
    "Fly_1_sex": "Fly_2_sex", "Fly_2_sex": "Fly_1_sex"
}).copy()
df_mirror["dc2"] = df_selected["dc2"]
if "dc1" in df_mirror.columns:
    df_mirror.drop(columns=["dc1"], inplace=True)

data_mirror = df_mirror.drop(columns=["B1"])

pc_mirror = RandomizedPC(data=data_mirror, alpha=fixed_alpha)
dag_mirror = pc_mirror.estimate()
model_mirror = DiscreteBayesianNetwork(dag_mirror.edges())

print("\nEdges in mirrored model:", list(model_mirror.edges()))

# Enforce dp ⊥ dc2
for u, v in [('dp', 'dc2'), ('dc2', 'dp')]:
    if model_mirror.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} violates dp ⊥ dc2. Removing.")
        removed_edges["mirrored"].append((u, v))
        model_mirror.remove_edge(u, v)

# Enforce Fly_1_sex ⊥ Fly_2_sex
for u, v in [('Fly_1_sex', 'Fly_2_sex'), ('Fly_2_sex', 'Fly_1_sex')]:
    if model_mirror.has_edge(u, v):
        print(f"⚠ Warning: Edge {u} -> {v} violates Fly_1_sex ⊥ Fly_2_sex. Removing.")
        removed_edges["mirrored"].append((u, v))
        model_mirror.remove_edge(u, v)

print("Edges after enforcing prior independencies (Mirrored):", list(model_mirror.edges()))

model_mirror.fit(data_mirror, estimator=MaximumLikelihoodEstimator)

plt.figure(figsize=(12, 8))
G_mirror = nx.DiGraph(model_mirror.edges())
G_mirror.add_nodes_from(data_mirror.columns)
pos_mirror = nx.spring_layout(G_mirror, seed=None)  # Remove fixed seed for layout variability
nx.draw(G_mirror, pos_mirror, with_labels=True, node_color="#c1f6e2", edge_color='black', node_size=3000, font_size=10, font_weight='bold')
plt.title("Mirrored Causal DAG with dc2 (PC Algorithm, B1 excluded)")
plt.show()

# === COMBINE NETWORKS ===
start_time_combined = time.time()
print("\n🔗 Combining Networks...")

# Get edges from both models
combined_edges = list(set(model.edges()).union(set(model_mirror.edges())))
G_combined = nx.DiGraph()
G_combined.add_edges_from(combined_edges)

# Use all nodes from both models
combined_nodes = set(model.nodes()).union(set(model_mirror.nodes()))
G_combined.add_nodes_from(combined_nodes)

# Generate layout and edge colors
pos_combined = nx.spring_layout(G_combined, seed=101)

edge_colors = []
for u, v in G_combined.edges():
    if (u, v) in model.edges() and (u, v) in model_mirror.edges():
        edge_colors.append('mediumpurple')  # present in both
    elif (u, v) in model.edges():
        edge_colors.append('skyblue')       # original only
    elif (u, v) in model_mirror.edges():
        edge_colors.append('salmon')        # mirrored only
    else:
        edge_colors.append('gray')

# Plotting the combined DAG
plt.figure(figsize=(16, 10))
nx.draw(
    G_combined,
    pos_combined,
    with_labels=True,
    node_color='lightgray',
    edge_color=edge_colors,
    node_size=3200,
    font_size=10,
    font_weight='bold'
)
plt.title("🔄 Final Combined Causal DAG (Full Original + Full Mirrored)", fontsize=14)
plt.tight_layout()
plt.show()

end_time_combined = time.time()
print(f"\n⏱ Execution Time (Combined Network): {end_time_combined - start_time_combined:.2f} seconds")

# === Final Summary ===
print("\n✅ Final Combined DAG Summary:")
print(f"🔹 Total unique edges: {len(combined_edges)}")
print("🔹 Edge color legend:")
print("   - 🔵 skyblue: Original only")
print("   - 🔴 salmon: Mirrored only")
print("   - 🟣 mediumpurple: Shared in both")

print("\n❌ Removed edges (due to prior knowledge constraints):")
print(f"Original: {removed_edges['original']}")
print(f"Mirrored: {removed_edges['mirrored']}")

# Total Execution Time
total_start_time = start_time_original
total_end_time = end_time_combined
print(f"\n⏱ Total Execution Time (All Parts): {total_end_time - total_start_time:.2f} seconds")