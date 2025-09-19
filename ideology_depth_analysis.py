import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14, "figure.figsize": (10,6)})

MODEL_1 = "gemma-2-9b-it_original"
# MODEL_2 = "llama-3.1-8b-it"
MODEL_2 = "gemma-2-9b-it_ablated"

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
    response_data = df[response_cols].T.fillna(0.5)
    fig, ax = plt.subplots(figsize=(10,4))
    label_map = {1: 'Liberal', 0.5: 'Null/Refusal',0: 'Conservative'}
    labels = response_data.applymap(lambda x: label_map.get(x, 'Null/Refusal'))
    colors = ['#a82d30', '#edf8b1', '#2175b2'] # blue to red gradient
    custom_cmap = ListedColormap(colors)  
    sns.heatmap(response_data, cmap=custom_cmap, #'RdBu', 
                center=0.5, xticklabels=False, ax=ax,
                # cbar_kws={'label': 'Response (0=Conservative, 1=Liberal)'}
                cbar_kws={'label': 'Response',
                          'ticks': [0, 0.5, 1], 
                          'format': lambda x, 
                          pos: label_map.get(x, 'Null/Refusal')},
                )
    plt.title(f'{model} Response Patterns')
    plt.xlabel('Questions')
    plt.ylabel('Conditions')
    plt.tight_layout()
    plt.savefig(f"analysis/{model}_response_heatmap.pdf", bbox_inches='tight')
    plt.savefig(f"analysis/{model}_response_heatmap.png", bbox_inches='tight')
    plt.clf()

def plot_conservative_tendency_by_category(df_list, models):
    # response_cols = [col for col in df_list[0].columns if col not in ['category', 'q_id']]
    # being conservative
    response_cols = ['role_conservative_argument_none', 'role_conservative_argument_liberal', 'role_conservative_argument_conservative']
    category_means1 = 1- df_list[0].groupby('category')[response_cols].mean().mean(axis=1)
    category_means2 = 1- df_list[1].groupby('category')[response_cols].mean().mean(axis=1)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        models[0]: category_means1,
        models[1]: category_means2
    })

    # Plot side-by-side bars
    ax = plot_data.plot(kind='bar', color=['#34A853', '#4285F4'], alpha=0.7, width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=11, rotation=45)
    # plt.title('Average Conservative Tendency by Topic - Comparison')
    # plt.xlabel('Liberal Score (0=Conservative, 1=Liberal)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,1.2)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=2)
    plt.tight_layout()
    plt.savefig("analysis/candidate_conservative_tendency_comparison.pdf")
    plt.savefig("analysis/candidate_conservative_tendency_comparison.png")
    plt.clf()

def plot_null_vote_tendency_by_category(df_list, models): 
    response_cols = [col for col in df_list[0].columns if col not in ['category', 'q_id']]
    # category_means1 = df_list[0].isna().groupby('category')[response_cols].sum().sum()
    # category_means2 = df_list[1].isna().groupby('category')[response_cols].sum().sum()
    # print(category_means1)
    # print(category_means2)
       
    categories = df_list[0].category.unique()
    null_rates = {}
    for df, candidate in zip(df_list, models):
        null_rates[candidate] = {}
            
        for category in categories:
            cat_data = df[df['category'] == category][response_cols]
            # print(f"total_questions {category} = ", len(cat_data))
            null_responses = int(cat_data.isna().sum().sum())
            
            null_rates[candidate][category] = null_responses/cat_data.size # need total votes
    
    # print(pd.DataFrame(null_rates))

    # Create DataFrame for plotting
    plot_data = pd.DataFrame(null_rates)
    plot_data =plot_data.loc[(plot_data!=0).any(axis=1)]

    # Plot side-by-side bars
    ax = plot_data.plot(kind='bar', color=['#34A853', '#4285F4'], alpha=0.7, width=0.8, figsize=(10,6))
    for container in ax.containers:
        heights = [patch.get_height() for patch in container.patches]
        labels = [f'{h:.2f}' if h > 0 else '' for h in heights]
        ax.bar_label(container, labels=labels, fontsize=11, rotation=45)
        
    plt.title('Null response rate by Topic - Comparison')
    plt.xlabel('Categories')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("analysis/candidate_null_vote_tendency_comparison_ablated.pdf")
    plt.savefig("analysis/candidate_null_vote_tendency_comparison_ablated.png")
    plt.clf()

def plot_topic_coherence_radar(topic_scores_list, labels, suffix):
    plt.subplot(projection='polar')
    colors = ['#34A853', '#4285F4']
    for idx, topic_scores in enumerate(topic_scores_list):
        categories = list(topic_scores.keys())#[:8]  # Limit for readability
        values = [topic_scores[cat] for cat in categories]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.plot(angles, values, 'o-', linewidth=2, color=colors[idx], alpha=0.7, label=labels[idx])
        plt.fill(angles, values, alpha=0.25, color=colors[idx])
        plt.xticks(angles[:-1], [cat[:15] + '...' if len(cat) > 15 else cat for cat in categories])
    # plt.title(f'{suffix.upper()} Topic-Specific Ideological Consistency')
    plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"analysis/candidate_topic_coherence_radar_{suffix}.pdf")
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
    plt.savefig(f"analysis/candidate_response_consistency_heatmap.pdf")
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
    plt.savefig(f"analysis/{model}_condition_ideological_clustering.pdf")
    plt.clf()

def calculate_topic_scores(df, suffix):
    """Calculate ideological consistency within each topic category"""
    
    def avg_correlation(response_cols, category_data):
        # Calculate average pairwise correlation within category
        correlations = []
        for i in range(len(response_cols)):
            for j in range(i+1, len(response_cols)):
                if len(category_data) > 1:
                    corr = category_data.iloc[:, i].corr(category_data.iloc[:, j])
                    if not np.isnan(corr):
                        correlations.append(corr)
        # {len(category_questions)} 
        return np.mean(correlations) if correlations else 0

    def cronbachs_alpha(response_cols, category_data):
        # Calculate Cronbach's Alpha (KR-20 approximation)
        n_items = len(response_cols)
        sum_variances = category_data.var(axis=0).sum()
        total_variance = category_data.sum(axis=1).var()
        
        #Handle zero variance
        if total_variance == 0:
            return  0.0

        alpha = (n_items / (n_items - 1)) * (1 - (sum_variances / total_variance))
        return alpha
    
    def get_fleiss_kappa(response_cols, category_data):
        """
        κ > 0.8: Almost perfect coherence
        0.6 < κ ≤ 0.8: Substantial coherence
        0.4 < κ ≤ 0.6: Moderate coherence
        0.2 < κ ≤ 0.4: Fair coherence
        κ ≤ 0.2: Poor coherence
        κ < 0: Systematic disagreement (worse than chance)
        """
        count_matrix, _ = aggregate_raters(category_data)
        # avoid 0 division -> nan for perfect agreement
        if np.std(count_matrix) == 0:
            return 1.0
        kappa = fleiss_kappa(count_matrix, method='fleiss')
        return kappa
        """ 
        # DOING IT MANUALLY
        N,n = category_data.shape  # Number of items (questions), Number of raters (roles)

        # Fleiss' Kappa is undefined if there are fewer than 2 raters or no items.
        assert n > 1 and N > 1

        # Step 1: Create a count matrix for how many raters assigned each item to each category.
        # For binary data [0, 1], we count the number of 1s (liberal) and 0s (conservative).
        n_liberal = category_data.sum(axis=1)
        n_conservative = n - n_liberal

        # The count_matrix has N rows (items) and 2 columns (categories).
        count_matrix = pd.concat([n_conservative, n_liberal], axis=1)

        # Step 2: Calculate P_i, the proportion of agreeing pairs for each item i.
        # Formula: P_i = (1 / (n * (n - 1))) * sum(n_ij * (n_ij - 1))
        # where n_ij is the number of raters who assigned item i to category j.
        sum_of_squares = (count_matrix ** 2).sum(axis=1)
        
        p_i = (sum_of_squares - n) / (n * (n - 1))

        # Step 3: Calculate P_bar, the mean of the P_i's (mean observed agreement).
        p_observed = p_i.mean()

        # Step 4: Calculate P_e, the expected agreement by chance.
        # First, calculate the proportion of all assignments belonging to each category.
        total_ratings = N * n
        p_j = count_matrix.sum(axis=0) / total_ratings

        # Then, sum the squares of these proportions.
        p_expected = (p_j ** 2).sum()

        # Step 5: Calculate Fleiss' Kappa.
        denominator = 1 - p_expected
        if denominator == 0:
            # If p_expected is 1, it means all raters always chose the same category distribution.
            # If observed agreement is also 1, then Kappa is 1. Otherwise, it is undefined.
            kappa = 1.0 if p_observed == 1.0 else float('nan')
        else:
            kappa = (p_observed - p_expected) / denominator
        return kappa
        """
    
    topic_scores = {}
    categories = df.category.unique()
    for category in categories:
        category_questions = df[df['category'] == category]
        if len(category_questions) < 5:
            continue
        
        # roleplaying
        # response_cols = [
        #     'role_original_argument_none', 'role_original_argument_liberal', 'role_original_argument_conservative'
        # ]
        # argument
        # response_cols = [
        #     'role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none'
        # ]
        
        if suffix=='role_original':
            # being original
            response_cols = [
                'role_original_argument_none', 'role_original_argument_liberal', 'role_original_argument_conservative'
            ]
        elif suffix=='role_liberal':
            # being liberal
            response_cols = [
                'role_liberal_argument_none', 'role_liberal_argument_liberal', 'role_liberal_argument_conservative'
            ]
        elif suffix=='role_conservative':
            # being conservative
            response_cols = [
                'role_conservative_argument_none', 'role_conservative_argument_liberal', 'role_conservative_argument_conservative'
            ]
        elif suffix=='steering_conservative_caa':
            # being conservative
            response_cols = [
                'steering_caa_argument_none', 'steering_caa_argument_liberal', 'steering_caa_argument_conservative'
            ]
        elif suffix=='steering_conservative_sta':
            # being conservative
            response_cols = [
                'steering_sta_argument_none', 'steering_sta_argument_liberal', 'steering_sta_argument_conservative'
            ]
        else:
            # across all conditions for this category
            response_cols = [col for col in df.columns if col not in ['category', 'q_id']]

        category_data = category_questions[response_cols]
        
        # topic_scores[f"{category}"] = avg_correlation(response_cols, category_data)
        # topic_scores[f"{category}"] = cronbachs_alpha(response_cols, category_data)
        topic_scores[f"{category}"] = get_fleiss_kappa(response_cols, category_data)

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
# df_1 = df_1.fillna(0.5)
df_2 = pd.read_csv(f"analysis/{MODEL_2}_labeled_votes.csv")
# df_2 = df_2.fillna(0.5)
# # print(df)

# plot_response_heatmap(df_1, MODEL_1)
# plot_response_heatmap(df_2, MODEL_2)

# plot_conservative_tendency_by_category([df_1, df_2], [MODEL_1, MODEL_2])

plot_null_vote_tendency_by_category([df_1, df_2], [MODEL_1, MODEL_2])

# topic_scores_1 = calculate_topic_scores(df_1, "role_all")
# topic_scores_2 = calculate_topic_scores(df_2, "role_all")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], "role_all")
# topic_scores_1 = calculate_topic_scores(df_1, "role_original")
# topic_scores_2 = calculate_topic_scores(df_2, "role_original")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], "role_original")
# topic_scores_1 = calculate_topic_scores(df_1, "role_liberal")
# topic_scores_2 = calculate_topic_scores(df_2, "role_liberal")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], "role_liberal")
# topic_scores_1 = calculate_topic_scores(df_1, "role_conservative")
# topic_scores_2 = calculate_topic_scores(df_2, "role_conservative")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], "role_conservative")

# intra_consistency_1 = calculate_intra_condition_consistency(df_1)
# intra_consistency_2 = calculate_intra_condition_consistency(df_2)
# # print(intra_consistency_1)
# # print(intra_consistency_2)
# plot_consistency_heatmap([intra_consistency_1, intra_consistency_2], [MODEL_1, MODEL_2])

# # plot_ideological_clustering(df_1, MODEL_1)
# # plot_ideological_clustering(df_2, MODEL_2)


# direction = "conservative"
# df_1 = pd.read_csv(f"analysis/{MODEL_1}_labeled_{direction}_steering_131K_votes.csv")
# df_2 = pd.read_csv(f"analysis/{MODEL_2}_labeled_{direction}_steering_131K_votes.csv")

# topic_scores_1 = calculate_topic_scores(df_1, f"steering_{direction}_caa")
# topic_scores_2 = calculate_topic_scores(df_2, f"steering_{direction}_caa")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], f"steering_131K_{direction}_caa")
# topic_scores_1 = calculate_topic_scores(df_1, f"steering_{direction}_sta")
# topic_scores_2 = calculate_topic_scores(df_2, f"steering_{direction}_sta")
# plot_topic_coherence_radar([topic_scores_1, topic_scores_2], [MODEL_1, MODEL_2], f"steering_131K_{direction}_sta")
