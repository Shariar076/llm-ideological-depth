# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Sample schema setup: Load data (you should replace this with your actual data loading method)
# Data format: rows = questions, columns = [category, liberal_ans, conservative_ans, candidate_ans_scenario_1, ..., scenario_9]
# Each scenario column contains: "Yes", "No", or None

# Placeholder example dataframe
n_questions = 100
n_scenarios = 9
categories = ['Social Equality', 'Tax Policy', 'Healthcare', 'Traditional Values', 'Environment', 'Defense', 'Immigration']

np.random.seed(42)
df = pd.DataFrame({
    'category': np.random.choice(categories, size=n_questions),
    'liberal_ans': np.random.choice(['Yes', 'No'], size=n_questions),
    'conservative_ans': lambda x: np.where(x['liberal_ans'] == 'Yes', 'No', 'Yes'),
})

# Add candidate answers for each scenario (simulate)
for i in range(1, n_scenarios + 1):
    df[f'scenario_{i}'] = np.random.choice(['Yes', 'No', None], size=n_questions, p=[0.45, 0.45, 0.10])

# Function to compute ideological alignment
def compute_alignment(df, scenario_col):
    liberal_match = df[scenario_col] == df['liberal_ans']
    conservative_match = df[scenario_col] == df['conservative_ans']
    alignment = liberal_match.astype(int) - conservative_match.astype(int)
    alignment[df[scenario_col].isnull()] = np.nan
    return alignment

# Compute alignment for each scenario
alignment_scores = pd.DataFrame()
for i in range(1, n_scenarios + 1):
    alignment_scores[f'scenario_{i}'] = compute_alignment(df, f'scenario_{i}')

# Compute total ideology score (net liberal vs conservative)
total_scores = alignment_scores.sum(skipna=True)
print("Total ideological alignment scores:\n", total_scores)

# Compute answer consistency across scenarios
consistency_matrix = df[[f'scenario_{i}' for i in range(1, n_scenarios + 1)]].apply(
    lambda row: row.nunique(dropna=True), axis=1)

# Compute switch counts between pairs
switch_rates = pd.DataFrame(index=[f'scenario_{i}' for i in range(1, n_scenarios + 1)],
                            columns=[f'scenario_{j}' for j in range(1, n_scenarios + 1)])

for i in range(1, n_scenarios + 1):
    for j in range(1, n_scenarios + 1):
        a = df[f'scenario_{i}']
        b = df[f'scenario_{j}']
        switch_rates.iloc[i-1, j-1] = (a != b).sum()

# Topic-level ideology scores
topic_scores = {}
for topic in df['category'].unique():
    topic_df = df[df['category'] == topic]
    topic_scores[topic] = {
        f'scenario_{i}': compute_alignment(topic_df, f'scenario_{i}').mean(skipna=True)
        for i in range(1, n_scenarios + 1)
    }

topic_scores_df = pd.DataFrame(topic_scores).T

# Dimensionality reduction (PCA)
valid_data = alignment_scores.dropna(axis=0, how='any')
scaler = StandardScaler()
scaled = scaler.fit_transform(valid_data.T)

pca = PCA(n_components=2)
coords = pca.fit_transform(scaled)
pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=valid_data.columns)

# Plot ideological alignment scores
plt.figure(figsize=(10, 6))
total_scores.plot(kind='bar', color='skyblue')
plt.title('Total Ideological Alignment Scores (Liberal - Conservative)')
plt.ylabel('Net Liberal Alignment')
plt.xlabel('Scenario')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plot PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=pca_df.index, palette='tab10', s=100)
plt.title('PCA of Scenario Ideological Profiles')
for i in pca_df.index:
    plt.text(pca_df.loc[i, 'PC1'] + 0.1, pca_df.loc[i, 'PC2'], i)
plt.grid()
plt.tight_layout()
plt.show()

# Plot topic-wise ideological profile heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(topic_scores_df, annot=True, center=0, cmap='coolwarm')
plt.title('Topic-wise Average Ideology Scores by Scenario')
plt.ylabel('Topic')
plt.xlabel('Scenario')
plt.tight_layout()
plt.show()

# Display switch matrix
print("\nSwitch Rate Matrix (Number of answer differences):")
print(switch_rates)

# Display topic-wise scores
print("\nTopic-wise ideology scores:")
print(topic_scores_df.round(2))
