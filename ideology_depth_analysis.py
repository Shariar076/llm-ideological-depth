import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14, "figure.figsize": (10,6)})

MODEL_1 = "gemma-2-9b-it"
MODEL_2 = "llama-3.1-8b-it"

def plot_response_heatmap(df, model):
    # response_cols = [col for col in df.columns if col not in ['category', 'q_id']]
    response_cols = [
            'role_original_argument_none', 'role_original_argument_liberal', 'role_original_argument_conservative',
            'role_liberal_argument_none', 'role_liberal_argument_liberal', 'role_liberal_argument_conservative',
            'role_conservative_argument_none', 'role_conservative_argument_liberal', 'role_conservative_argument_conservative'
            # 'role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none', 
            # 'role_original_argument_liberal', 'role_liberal_argument_liberal', 'role_conservative_argument_liberal',
            # 'role_original_argument_conservative', 'role_liberal_argument_conservative', 'role_conservative_argument_conservative'
        ]
    response_data = df[response_cols].T
    sns.heatmap(response_data, cmap='RdBu', 
                center=0.5, xticklabels=False,
                cbar_kws={'label': 'Response (0=Conservative, 1=Liberal)'})
    plt.title('Response Patterns Across All Conditions')
    plt.xlabel('Questions')
    plt.ylabel('Conditions')
    plt.tight_layout()
    plt.savefig(f"analysis/{model}_response_heatmap.png")
    plt.clf()

def plot_liberal_tendency_by_category(df_list, models):
    response_cols = [col for col in df_list[0].columns if col not in ['category', 'q_id']]
    category_means1 = df_list[0].groupby('category')[response_cols].mean().mean(axis=1)
    category_means2 = df_list[1].groupby('category')[response_cols].mean().mean(axis=1)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        models[0]: category_means1,
        models[1]: category_means2
    })

    # Plot side-by-side bars
    ax = plot_data.plot(kind='bar', color=['orange', 'skyblue'], alpha=0.7, width=0.8)
    plt.title('Average Liberal Tendency by Topic - Comparison')
    plt.xlabel('Liberal Score (0=Conservative, 1=Liberal)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,1)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("analysis/candidate_liberal_tendency_comparison.png")
    plt.clf()

def plot_null_vote_tendency_by_category(df_list, models, null_value=0.5):
    response_cols = [col for col in df_list[0].columns if col not in ['category', 'q_id']]
    # category_means1 = (df_list[0]==null_value).groupby('category')[response_cols].sum().sum()
    # category_means2 = (df_list[1]==null_value).groupby('category')[response_cols].sum().sum()
    # print(category_means1)
    # print(category_means2)
       
    categories = df_list[0].category.unique()
    null_rates = {}
    for df, candidate in zip(df_list, models):
        null_rates[candidate] = {}
            
        for category in categories:
            cat_data = df[df['category'] == category][response_cols]
            # print(f"total_questions {category} = ", len(cat_data))
            null_responses = int((cat_data==null_value).sum().sum())
            
            null_rates[candidate][category] = null_responses/cat_data.size # need total votes
    
    # print(pd.DataFrame(null_rates))

    # Create DataFrame for plotting
    plot_data = pd.DataFrame(null_rates)
    plot_data =plot_data.loc[(plot_data!=0).any(axis=1)]

    # Plot side-by-side bars
    ax = plot_data.plot(kind='bar', color=['orange', 'salmon'], alpha=0.7, width=0.8, figsize=(7,6))
    plt.title('Null response rate by Topic - Comparison')
    plt.xlabel('Categories')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("analysis/candidate_null_vote_tendency_comparison.png")
    plt.clf()

def plot_topic_coherence_radar(topic_scores_list, labels, suffix):
    plt.subplot(projection='polar')
    colors = ['red', 'blue', 'green', 'orange']
    for idx, topic_scores in enumerate(topic_scores_list):
        categories = list(topic_scores.keys())#[:8]  # Limit for readability
        values = [topic_scores[cat] for cat in categories]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.plot(angles, values, 'o-', linewidth=2, color=colors[idx], alpha=0.7, label=labels[idx])
        plt.fill(angles, values, alpha=0.25, color=colors[idx])
        plt.xticks(angles[:-1], [cat[:15] + '...' if len(cat) > 15 else cat for cat in categories])
    plt.title('Topic-Specific Ideological Consistency')
    plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"analysis/candidate_topic_coherence_radar_{suffix}.png")
    plt.clf()

def plot_consistency_heatmap(consistencies_list, labels):
    conditions_values = consistencies_list[0].keys()
    consistency_matrix = []
    for consistencies in consistencies_list:
        row = []
        for condition in conditions_values:
            row.append(consistencies[condition])
        consistency_matrix.append(row)
    
    sns.heatmap(consistency_matrix, 
                xticklabels=list(conditions_values),
                yticklabels=[f'{c}' for c in labels],
                annot=True, fmt='.2f', cmap='RdYlBu_r')
    plt.title('Consistency Across Conditions')
    plt.xticks(rotation=45,  ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"analysis/candidate_response_consistency_heatmap.png")
    plt.clf()

def plot_ideological_clustering(df, model):
    response_cols = [col for col in df.columns if col not in ['category', 'q_id']]
    
    # PCA for dimensionality reduction
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[response_cols].T)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=1.0)
    for i, condition in enumerate(response_cols):
        plt.annotate(condition, (pca_data[i, 0], pca_data[i, 1]), fontsize=12, rotation=0)
    plt.title(f'Ideological Position Clustering (PCA) {model}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.tight_layout()
    plt.savefig(f"analysis/{model}_condition_ideological_clustering.png")
    plt.clf()

def calculate_topic_scores(df):
    """Calculate ideological consistency within each topic category"""
    topic_scores = {}
    categories = df.category.unique()
    for category in categories:
        category_questions = df[df['category'] == category]
        if len(category_questions) < 3:
            continue
        
        # roleplaying
        # response_cols = [
        #     'role_original_argument_none', 'role_original_argument_liberal', 'role_original_argument_conservative'
        # ]
        # argument
        # response_cols = [
        #     'role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none'
        # ]

        # across all conditions for this category
        response_cols = [col for col in df.columns if col not in ['category', 'q_id']]

        category_data = category_questions[response_cols]
        
        # Calculate average pairwise correlation within category
        correlations = []
        for i in range(len(response_cols)):
            for j in range(i+1, len(response_cols)):
                if len(category_data) > 1:
                    corr = category_data.iloc[:, i].corr(category_data.iloc[:, j])
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        topic_scores[category] = np.mean(correlations) if correlations else 0
    
    return topic_scores

def calculate_intra_condition_consistency(df):
    response_cols = [col for col in df.columns if col not in ['category', 'q_id']]
    intra_consistency = {}
    for condition in response_cols:
        responses = df[condition]
        # Measure how consistent responses are (closer to 0 or 1)
        # 4 is a scaling factor. Since responses are binary (0 or 1), the maximum possible variance is 0.25 
        # (when responses are perfectly split 50/50). Multiplying by 4 scales this to a 0-1 range
        intra_consistency[condition] = 1 - (4 * responses.var())
    return intra_consistency
        
df_1 = pd.read_csv(f"analysis/{MODEL_1}_labeled_votes.csv")
df_1 = df_1.fillna(0.5)
df_2 = pd.read_csv(f"analysis/{MODEL_2}_labeled_votes.csv")
df_2 = df_2.fillna(0.5)

# print(df)

# plot_response_heatmap(df_1, MODEL_1)
# plot_response_heatmap(df_2, MODEL_2)

# plot_liberal_tendency_by_category([df_1, df_2], [MODEL_1, MODEL_2])

plot_null_vote_tendency_by_category([df_1, df_2], [MODEL_1, MODEL_2])

# topic_scores_1 = calculate_topic_scores(df_1)
# topic_scores_2 = calculate_topic_scores(df_2)
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], "all")

# intra_consistency_1 = calculate_intra_condition_consistency(df_1)
# intra_consistency_2 = calculate_intra_condition_consistency(df_2)
# # print(intra_consistency_1)
# # print(intra_consistency_2)
# plot_consistency_heatmap([intra_consistency_1, intra_consistency_2], [MODEL_1, MODEL_2])

# plot_ideological_clustering(df_1, MODEL_1)
# plot_ideological_clustering(df_2, MODEL_2)