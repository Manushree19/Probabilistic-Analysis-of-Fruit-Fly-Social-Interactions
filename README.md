# 🧬 Probabilistic Analysis of Fruit Fly Social Interactions

### 📘 Overview
This project models and predicts the social behavior of *Drosophila melanogaster* (fruit flies) using **probabilistic modeling** and **Dynamic Bayesian Networks (DBNs)**.  
It explores how different pairs of fruit flies (male–male, female–female, and courtship pairs) synchronize their behavior and influence each other over time.

---

### 🎯 Objectives
- Analyze behavioral correlations using real datasets.
- Model behavioral transitions using **Dynamic Bayesian Networks**.
- Infer causal relationships through the **PC Algorithm** and its **Randomized variant**.
- Visualize synchronization with **heatmaps** generated in MATLAB.

---

### 🧠 Background
Fruit flies are ideal model organisms for studying social behavior due to their simple nervous systems and well-known genetics.  
Using probabilistic and computational methods, this project links observed behavioral data to mathematical models that explain **coordination, cooperation, and competition**.

---

### 🧮 Methodology
#### 1. Data Analysis
- Datasets: `courtship_complete.csv`, `male_complete.csv`, `female_complete.csv`
- MATLAB functions calculate Pearson’s correlation between fly behaviors.

#### 2. Visualization
- Heatmaps show how behaviors of Fly 1 and Fly 2 align over time.
- Diagonal dominance = synchronized behavior.

#### 3. Algorithms
- **PC Algorithm:** Deterministic baseline for Bayesian structure learning.
- **Randomized PC Algorithm:** Introduces random variable order and bootstrap sampling for robustness.
- Both output **Directed Acyclic Graphs (DAGs)** representing behavioral dependencies.

---

### 🧪 Results
| Pair Type | Observation |
|------------|--------------|
| Courtship (Male–Female) | Strong synchronization due to courtship signaling |
| Male–Male | Moderate coordination, competitive tendencies |
| Female–Female | Unexpectedly high synchronization and cooperation |

---

### ⚙️ How to Run

#### Requirements
- MATLAB R2022a or newer  
- Statistics and Machine Learning Toolbox  

#### Steps
```matlab
% Load datasets
courtshipData = readtable('courtship_complete.csv');
maleData = readtable('male_complete.csv');
femaleData = readtable('female_complete.csv');

% Run the analysis
run('main.m');
