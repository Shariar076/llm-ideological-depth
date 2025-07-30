import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PoliticalIdeologyAnalyzer:
    def __init__(self):
        self.categories = {
            'Political & Ideological Stances': 9,
            'Tax Policy': 1,
            'Healthcare': 2,
            'Abortion Rights': 1,
            'Social Equality & Civil Rights': 19,
            'LGBTQ+ Rights': 2,
            'Social Welfare & Poverty': 29,
            'Corporate & Economic Regulation': 5,
            'Climate & Environment': 1,
            'Immigration & Refugees': 1,
            'Military & Defense Spending': 2,
            'Gun Control': 1,
            'Traditional Values & Gender Roles': 27
        }
        
        # Create question-to-category mapping
        self.question_categories = {}
        q_idx = 0
        for category, count in self.categories.items():
            for i in range(count):
                self.question_categories[f'Q{q_idx+1}'] = category
                q_idx += 1
    
    def load_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)  # for reproducibility
        
        # Create sample responses (1 = liberal, 0 = conservative)
        data = {}
        
        # Generate responses for each condition
        conditions = [
            # 'base_original', 'base_liberal', 'base_conservative',
            # 'lib_framed_original', 'lib_framed_liberal', 'lib_framed_conservative',
            # 'cons_framed_original', 'cons_framed_liberal', 'cons_framed_conservative'
            'role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none', 
            'role_original_argument_liberal', 'role_liberal_argument_liberal', 'role_conservative_argument_liberal',
            'role_original_argument_conservative', 'role_liberal_argument_conservative', 'role_conservative_argument_conservative'
        ]
        
        for condition in conditions:
            # Simulate responses with some ideological consistency
            if 'role_liberal' in condition:
                base_prob = 0.8  # More likely to give liberal answers
            elif 'role_conservative' in condition:
                base_prob = 0.2  # More likely to give conservative answers
            else:  # original persona
                base_prob = 0.6  # Slightly liberal-leaning
            
            # Add some noise and framing effects
            if 'argument_liberal' in condition:
                base_prob += 0.1  # Liberal framing pushes toward liberal
            elif 'argument_conservative' in condition:
                base_prob -= 0.1  # Conservative framing pushes toward conservative
            
            base_prob = np.clip(base_prob, 0.1, 0.9)
            
            responses = np.random.binomial(1, base_prob, 100)
            data[condition] = responses
        
        # Add question categories
        categories = []
        q_idx = 0
        for category, count in self.categories.items():
            categories.extend([category] * count)
        
        data['category'] = categories
        data['question'] = [f'Q{i+1}' for i in range(100)]
        
        return pd.DataFrame(data)
    
    def calculate_consistency_scores(self, df):
        """Calculate various consistency metrics"""
        results = {}
        
        # 1. Within-persona consistency (across framing conditions)
        personas = ['role_original', 'role_liberal', 'role_conservative']
        for persona in personas:
            cols = [col for col in df.columns if persona in col]
            if len(cols) >= 2:
                # Calculate pairwise correlations
                correlations = []
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        corr, _ = pearsonr(df[cols[i]], df[cols[j]])
                        if not np.isnan(corr):
                            correlations.append(corr)
                results[f'{persona}_consistency'] = np.mean(correlations) if correlations else 0
        
        # 2. Cross-persona flexibility
        base_conditions = ['role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none']
        if all(col in df.columns for col in base_conditions):
            # Measure how different the personas are from each other
            lib_cons_diff = np.mean(np.abs(df['role_liberal_argument_none'] - df['role_conservative_argument_none']))
            orig_range = (np.mean(np.abs(df['role_original_argument_none'] - df['role_liberal_argument_none'])) + 
                         np.mean(np.abs(df['role_original_argument_none'] - df['role_conservative_argument_none']))) / 2
            results['persona_flexibility'] = lib_cons_diff
            results['original_distinctiveness'] = orig_range
        
        # 3. Resistance to framing
        framing_resistance = {}
        for persona in personas:
            base_col = f'{persona}_argument_none'
            lib_framed_col = f'{persona}_argument_liberal'
            cons_framed_col = f'{persona}_argument_conservative'
            
            if all(col in df.columns for col in [base_col, lib_framed_col, cons_framed_col]):
                lib_resistance = 1 - np.mean(np.abs(df[base_col] - df[lib_framed_col]))
                cons_resistance = 1 - np.mean(np.abs(df[base_col] - df[cons_framed_col]))
                framing_resistance[persona] = (lib_resistance + cons_resistance) / 2
        
        results['framing_resistance'] = framing_resistance
        
        return results
    
    def calculate_topic_expertise(self, df):
        """Calculate ideological consistency within each topic category"""
        topic_scores = {}
        
        for category in self.categories.keys():
            category_questions = df[df['category'] == category]
            if len(category_questions) < 2:
                print("XXXXXXXXXXXXX",category, len(category_questions))
                continue
            
            # Calculate consistency across all conditions for this category
            response_cols = [col for col in df.columns if col not in ['category', 'question']]
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
    
    def create_visualizations(self, df, consistency_scores, topic_expertise):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Heatmap of responses across conditions
        plt.subplot(3, 3, 1)
        response_cols = [col for col in df.columns if col not in ['category', 'q_id']]
        response_data = df[response_cols].T
        print(response_data)
        sns.heatmap(response_data, cmap='RdBu_r', center=0.5, cbar_kws={'label': 'Response (0=Conservative, 1=Liberal)'})
        plt.title('Response Patterns Across All Conditions')
        plt.xlabel('Questions')
        plt.ylabel('Conditions')
        
        # 2. Consistency scores bar plot
        plt.subplot(3, 3, 2)
        consistency_items = [(k, v) for k, v in consistency_scores.items() if isinstance(v, (int, float))]
        if consistency_items:
            keys, values = zip(*consistency_items)
            plt.bar(keys, values, color='skyblue', alpha=0.7)
            plt.title('Ideological Consistency Scores')
            plt.xticks(rotation=45)
            plt.ylabel('Consistency Score')
        
        # 3. Topic expertise radar chart
        plt.subplot(3, 3, 3, projection='polar')
        if topic_expertise:
            categories = list(topic_expertise.keys())#[:8]  # Limit for readability
            values = [topic_expertise[cat] for cat in categories]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            plt.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
            plt.fill(angles, values, alpha=0.25, color='red')
            plt.xticks(angles[:-1], [cat[:15] + '...' if len(cat) > 15 else cat for cat in categories])
            plt.title('Topic-Specific Ideological Consistency')
        
        # 4. Framing resistance comparison
        plt.subplot(3, 3, 4)
        if 'framing_resistance' in consistency_scores:
            personas = list(consistency_scores['framing_resistance'].keys())
            resistance_values = list(consistency_scores['framing_resistance'].values())
            plt.bar(personas, resistance_values, color=['blue', 'red', 'green'], alpha=0.7)
            plt.title('Resistance to Question Framing')
            plt.ylabel('Resistance Score (Higher = More Resistant)')
        
        # 5. Response distribution by category
        plt.subplot(3, 3, 5)
        category_means = df.groupby('category')[response_cols].mean().mean(axis=1)
        category_means.plot(kind='barh', color='orange', alpha=0.7)
        plt.title('Average Liberal Tendency by Topic')
        plt.xlabel('Liberal Score (0=Conservative, 1=Liberal)')
        
        # 6. Persona comparison
        plt.subplot(3, 3, 6)
        persona_cols = ['role_original_argument_none', 'role_liberal_argument_none', 'role_conservative_argument_none']
        if all(col in df.columns for col in persona_cols):
            persona_means = df[persona_cols].mean()
            plt.bar(range(len(persona_means)), persona_means, 
                   color=['purple', 'blue', 'red'], alpha=0.7)
            plt.xticks(range(len(persona_means)), ['Original', 'Liberal Role', 'Conservative Role'])
            plt.title('Average Response by Persona')
            plt.ylabel('Liberal Score')
        
        # 7. Question difficulty (variance across personas)
        plt.subplot(3, 3, 7)
        if all(col in df.columns for col in persona_cols):
            question_variance = df[persona_cols].var(axis=1)
            plt.hist(question_variance, bins=20, alpha=0.7, color='green')
            plt.title('Question Difficulty Distribution')
            plt.xlabel('Response Variance Across Personas')
            plt.ylabel('Number of Questions')
        
        # 8. Ideological clustering
        plt.subplot(3, 3, 8)
        if len(response_cols) >= 3:
            # PCA for dimensionality reduction
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[response_cols].T)
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
            for i, condition in enumerate(response_cols):
                plt.annotate(condition, (pca_data[i, 0], pca_data[i, 1]), fontsize=8)
            plt.title('Ideological Position Clustering (PCA)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # 9. Summary metrics
        plt.subplot(3, 3, 9)
        plt.axis('off')
        summary_text = "IDEOLOGICAL DEPTH SUMMARY\n\n"
        
        # Calculate overall scores
        if consistency_items:
            avg_consistency = np.mean([v for k, v in consistency_items])
            summary_text += f"Average Consistency: {avg_consistency:.3f}\n"
        
        if topic_expertise:
            max_expertise = max(topic_expertise.values())
            min_expertise = min(topic_expertise.values())
            summary_text += f"Strongest Topic: {max_expertise:.3f}\n"
            summary_text += f"Weakest Topic: {min_expertise:.3f}\n"
        
        if 'persona_flexibility' in consistency_scores:
            summary_text += f"Persona Flexibility: {consistency_scores['persona_flexibility']:.3f}\n"
        
        plt.text(0.1, 0.7, summary_text, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        # plt.show()
        plt.savefig('fig-1.png')
    
    def generate_report(self, df, consistency_scores, topic_expertise):
        """Generate a comprehensive text report"""
        print("="*60)
        print("POLITICAL IDEOLOGICAL DEPTH ANALYSIS REPORT")
        print("="*60)
        
        print("\n1. OVERALL CONSISTENCY METRICS:")
        print("-" * 30)
        for metric, value in consistency_scores.items():
            if isinstance(value, dict):
                print(f"{metric.upper()}:")
                for sub_metric, sub_value in value.items():
                    print(f"  {sub_metric}: {sub_value:.3f}")
            else:
                print(f"{metric}: {value:.3f}")
        
        print("\n2. TOPIC-SPECIFIC EXPERTISE:")
        print("-" * 30)
        sorted_topics = sorted(topic_expertise.items(), key=lambda x: x[1], reverse=True)
        for topic, score in sorted_topics:
            print(f"{topic}: {score:.3f}")
        
        print("\n3. IDEOLOGICAL PROFILE INTERPRETATION:")
        print("-" * 30)
        
        # Interpret consistency scores
        avg_consistency = np.mean([v for k, v in consistency_scores.items() 
                                 if isinstance(v, (int, float))])
        if avg_consistency > 0.7:
            print("• HIGH ideological consistency - well-structured belief system")
        elif avg_consistency > 0.4:
            print("• MODERATE ideological consistency - some structure with flexibility")
        else:
            print("• LOW ideological consistency - fragmented or evolving beliefs")
        
        # Interpret topic expertise
        topic_range = max(topic_expertise.values()) - min(topic_expertise.values())
        if topic_range > 0.3:
            print("• SPECIALIZED expertise - strong knowledge in specific domains")
        else:
            print("• GENERALIZED knowledge - consistent across all topics")
        
        # Framing resistance
        if 'framing_resistance' in consistency_scores:
            avg_resistance = np.mean(list(consistency_scores['framing_resistance'].values()))
            if avg_resistance > 0.8:
                print("• HIGH resistance to framing effects - strong convictions")
            elif avg_resistance > 0.6:
                print("• MODERATE resistance to framing effects")
            else:
                print("• LOW resistance to framing effects - malleable opinions")
        
        print("\n4. RECOMMENDATIONS FOR FURTHER ANALYSIS:")
        print("-" * 30)
        print("• Examine questions with highest variance for potential issues")
        print("• Compare results with established political scales")
        print("• Consider qualitative analysis of reasoning patterns")
        print("• Test with additional framing conditions")

# Example usage
def main():
    analyzer = PoliticalIdeologyAnalyzer()
    
    # Load sample data (replace with your actual data loading)
    # df = analyzer.load_sample_data()
    df = pd.read_csv("analysis/llama-3.1-8b-it_labeled_votes.csv")
    df = df.fillna(0.5)
    print(df)
    print("Sample data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Calculate metrics
    consistency_scores = analyzer.calculate_consistency_scores(df)
    topic_expertise = analyzer.calculate_topic_expertise(df)
    
    # Create visualizations
    analyzer.create_visualizations(df, consistency_scores, topic_expertise)
    
    # Generate report
    analyzer.generate_report(df, consistency_scores, topic_expertise)

if __name__ == "__main__":
    main()