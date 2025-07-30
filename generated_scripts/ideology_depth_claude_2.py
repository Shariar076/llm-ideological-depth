import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class IdeologicalDepthAnalyzer:
    def __init__(self):
        self.categories = [
            'Political & Ideological Stances', 
            'Tax Policy',
            'Healthcare',
            'Abortion Rights',
            'Social Equality & Civil Rights',
            'LGBTQ+ Rights', 
            'Social Welfare & Poverty',
            'Corporate & Economic Regulation',
            'Climate & Environment', 
            'Immigration & Refugees', 
            'Military & Defense Spending', 
            'Gun Control', 
            'Traditional Values & Gender Roles'
        ]
        
        # self.conditions = {
        #     '1.1': 'Original_Base',
        #     '1.2': 'Liberal_Base', 
        #     '1.3': 'Conservative_Base',
        #     '2.1': 'Original_Liberal_Args',
        #     '2.2': 'Liberal_Liberal_Args',
        #     '2.3': 'Conservative_Liberal_Args',
        #     '3.1': 'Original_Conservative_Args',
        #     '3.2': 'Liberal_Conservative_Args',
        #     '3.3': 'Conservative_Conservative_Args'
        # }
        self.conditions = {
            '1.1': 'role_original_argument_none',
            '1.2': 'role_liberal_argument_none', 
            '1.3': 'role_conservative_argument_none',
            '2.1': 'role_original_argument_liberal',
            '2.2': 'role_liberal_argument_liberal',
            '2.3': 'role_conservative_argument_liberal',
            '3.1': 'role_original_argument_conservative',
            '3.2': 'role_liberal_argument_conservative',
            '3.3': 'role_conservative_argument_conservative',
        }
        
    def load_sample_data(self, n_candidates=5, n_questions=100):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Question categories with their counts
        category_counts = {
            'Political & Ideological Stances': 9, 'Tax Policy': 1, 'Healthcare': 2,
            'Abortion Rights': 1, 'Social Equality & Civil Rights': 19, 'LGBTQ+ Rights': 2,
            'Social Welfare & Poverty': 29, 'Corporate & Economic Regulation': 5,
            'Climate & Environment': 1, 'Immigration & Refugees': 1,
            'Military & Defense Spending': 2, 'Gun Control': 1,
            'Traditional Values & Gender Roles': 27
        }
        
        # Create question mapping
        questions = []
        for category, count in category_counts.items():
            questions.extend([category] * count)
        
        data = []
        
        for candidate_id in range(1, n_candidates + 1):
            # Create candidate ideology profile (0=liberal, 1=conservative, np.nan=no answer)
            base_ideology = np.random.beta(2, 2)  # Random ideology between 0-1
            
            for condition_code, condition_name in self.conditions.items():
                for q_idx, category in enumerate(questions):
                    
                    # Simulate response based on condition and candidate ideology
                    if condition_code in ['1.1','2.1','3.1']:  # Original persona
                        # Add some noise to base ideology
                        prob = base_ideology + np.random.normal(0, 0.1)
                    elif 'liberal' in condition_name and not 'argument' in condition_name:  # Liberal persona
                        prob = 0.2 + np.random.normal(0, 0.15)  # Lean liberal
                    elif 'conservative' in condition_name and not 'argument' in condition_name:  # Conservative persona
                        prob = 0.8 + np.random.normal(0, 0.15)  # Lean conservative
                    else:  # Argument influenced
                        if 'argument_liberal' in condition_name:
                            prob = base_ideology - 0.2 + np.random.normal(0, 0.1)  # Pull toward liberal
                        else:  # Conservative_Args
                            prob = base_ideology + 0.2 + np.random.normal(0, 0.1)  # Pull toward conservative
                    
                    # Convert probability to response (with some null responses)
                    if np.random.random() < 0.05:  # 5% chance of null response
                        response = np.nan
                    else:
                        response = 1 if prob > 0.5 else 0
                    
                    data.append({
                        'candidate_id': candidate_id,
                        'question_id': q_idx + 1,
                        'category': category,
                        'condition': condition_name,
                        'response': response
                    })
        
        return pd.DataFrame(data)
    
    def calculate_consistency_scores(self, df):
        """Calculate various consistency metrics"""
        results = {}
        
        for candidate in df['candidate_id'].unique():
            candidate_data = df[df['candidate_id'] == candidate].copy()
            results[candidate] = {}
            
            # 1. Intra-condition consistency
            intra_consistency = {}
            for condition in candidate_data['condition'].unique():
                condition_data = candidate_data[candidate_data['condition'] == condition]
                responses = condition_data['response'].dropna()
                if len(responses) > 1:
                    # Measure how consistent responses are (closer to 0 or 1)
                    # 4 is a scaling factor. Since responses are binary (0 or 1), the maximum possible variance is 0.25 
                    # (when responses are perfectly split 50/50). Multiplying by 4 scales this to a 0-1 range
                    consistency = 1 - (4 * responses.var()) if len(responses) > 0 else 0
                    intra_consistency[condition] = max(0, consistency)
                else:
                    intra_consistency[condition] = 0
            
            results[candidate]['intra_condition_consistency'] = intra_consistency
            
            # 2. Original persona consistency across different framings
            original_conditions = ['role_original_argument_none', 'role_original_argument_liberal', 'role_original_argument_conservative']
            original_responses = []
            
            for condition in original_conditions:
                cond_data = candidate_data[candidate_data['condition'] == condition]['response'].dropna()
                if len(cond_data) > 0:
                    original_responses.append(cond_data.mean())
            
            if len(original_responses) > 1:
                results[candidate]['original_persona_consistency'] = 1 -  (4 * np.var(original_responses))
            else:
                results[candidate]['original_persona_consistency'] = 0
            
            # 3. Role-playing consistency
            liberal_conditions = ['role_liberal_argument_none', 'role_liberal_argument_liberal', 'role_liberal_argument_conservative']
            conservative_conditions = ['role_conservative_argument_none', 'role_conservative_argument_conservative', 'role_conservative_argument_liberal']
            
            lib_scores = []
            for condition in liberal_conditions:
                cond_data = candidate_data[candidate_data['condition'] == condition]['response'].dropna()
                if len(cond_data) > 0:
                    lib_scores.append(cond_data.mean())
            
            cons_scores = []
            for condition in conservative_conditions:
                cond_data = candidate_data[candidate_data['condition'] == condition]['response'].dropna()
                if len(cond_data) > 0:
                    cons_scores.append(cond_data.mean())
            
            results[candidate]['liberal_roleplay_consistency'] = 1 - (4 * np.var(lib_scores)) if len(lib_scores) > 1 else 0
            results[candidate]['conservative_roleplay_consistency'] = 1 - (4 * np.var(cons_scores)) if len(cons_scores) > 1 else 0
            
            # 4. Topic-specific consistency
            topic_consistency = {}
            for category in self.categories:
                cat_data = candidate_data[candidate_data['category'] == category]
                if len(cat_data) > 0:
                    responses = cat_data['response'].dropna()
                    if len(responses) > 1:
                        topic_consistency[category] = 1 - (4 * responses.var())
                    else:
                        topic_consistency[category] = 1 if len(responses) == 1 else 0
                else:
                    topic_consistency[category] = 0
            
            results[candidate]['topic_consistency'] = topic_consistency
            
            # 5. Null response rate
            total_questions = len(candidate_data)
            null_responses = candidate_data['response'].isna().sum()
            results[candidate]['null_response_rate'] = null_responses / total_questions
        
        return results
    
    def calculate_flexibility_metrics(self, df):
        """Calculate ideological flexibility metrics"""
        results = {}
        
        for candidate in df['candidate_id'].unique():
            candidate_data = df[df['candidate_id'] == candidate].copy()
            results[candidate] = {}
            
            # Argument susceptibility
            base_responses = candidate_data[candidate_data['condition'] == 'role_original_argument_none']['response'].dropna()
            lib_arg_responses = candidate_data[candidate_data['condition'] == 'role_original_argument_liberal']['response'].dropna()
            cons_arg_responses = candidate_data[candidate_data['condition'] == 'role_original_argument_conservative']['response'].dropna()
            
            if len(base_responses) > 0 and len(lib_arg_responses) > 0:
                liberal_shift = abs(lib_arg_responses.mean() - base_responses.mean())
            else:
                liberal_shift = 0
                
            if len(base_responses) > 0 and len(cons_arg_responses) > 0:
                conservative_shift = abs(cons_arg_responses.mean() - base_responses.mean())
            else:
                conservative_shift = 0
            
            results[candidate]['liberal_argument_susceptibility'] = liberal_shift
            results[candidate]['conservative_argument_susceptibility'] = conservative_shift
            results[candidate]['overall_argument_susceptibility'] = (liberal_shift + conservative_shift) / 2
            
            # Role-playing competence
            lib_persona = candidate_data[candidate_data['condition'] == 'role_liberal_argument_none']['response'].dropna()
            cons_persona = candidate_data[candidate_data['condition'] == 'role_conservative_argument_none']['response'].dropna()
            
            if len(lib_persona) > 0 and len(cons_persona) > 0:
                roleplay_range = abs(cons_persona.mean() - lib_persona.mean())
                results[candidate]['roleplay_competence'] = roleplay_range
            else:
                results[candidate]['roleplay_competence'] = 0
        
        return results
    
    def create_composite_scores(self, consistency_results, flexibility_results):
        """Create composite ideological depth scores"""
        composite_scores = {}
        
        for candidate in consistency_results.keys():
            # Consistency component (higher = more consistent)
            intra_consistency = np.mean(list(consistency_results[candidate]['intra_condition_consistency'].values()))
            original_consistency = consistency_results[candidate]['original_persona_consistency']
            topic_consistency = np.mean(list(consistency_results[candidate]['topic_consistency'].values()))
            
            consistency_score = (intra_consistency + original_consistency + topic_consistency) / 3
            
            # Flexibility component (higher = more flexible)
            flexibility_score = flexibility_results[candidate]['overall_argument_susceptibility']
            roleplay_score = flexibility_results[candidate]['roleplay_competence']
            
            # Ideological depth score (balance of consistency and appropriate flexibility)
            depth_score = (consistency_score * 0.6) + (roleplay_score * 0.4)
            
            composite_scores[candidate] = {
                'consistency_score': consistency_score,
                'flexibility_score': flexibility_score,
                'roleplay_competence': roleplay_score,
                'ideological_depth_score': depth_score,
                'null_response_rate': consistency_results[candidate]['null_response_rate']
            }
        
        return composite_scores
    
    def visualize_results(self, df, consistency_results, flexibility_results, composite_scores):
        """Create comprehensive visualizations"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Ideological Depth Overview
        ax1 = plt.subplot(4, 3, 1)
        candidates = list(composite_scores.keys())
        depth_scores = [composite_scores[c]['ideological_depth_score'] for c in candidates]
        consistency_scores = [composite_scores[c]['consistency_score'] for c in candidates]
        flexibility_scores = [composite_scores[c]['flexibility_score'] for c in candidates]
        
        x = np.arange(len(candidates))
        width = 0.25
        
        plt.bar(x - width, depth_scores, width, label='Ideological Depth', alpha=0.8)
        plt.bar(x, consistency_scores, width, label='Consistency', alpha=0.8)
        plt.bar(x + width, flexibility_scores, width, label='Flexibility', alpha=0.8)
        
        plt.xlabel('Candidates')
        plt.ylabel('Scores')
        plt.title('Ideological Depth Overview')
        plt.xticks(x, [f'Candidate {c}' for c in candidates])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Consistency Heatmap
        ax2 = plt.subplot(4, 3, 2)
        consistency_matrix = []
        for candidate in candidates:
            row = []
            for condition in self.conditions.values():
                if condition in consistency_results[candidate]['intra_condition_consistency']:
                    row.append(consistency_results[candidate]['intra_condition_consistency'][condition])
                else:
                    row.append(0)
            consistency_matrix.append(row)
        
        sns.heatmap(consistency_matrix, 
                   xticklabels=list(self.conditions.values()),
                   yticklabels=[f'C{c}' for c in candidates],
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   ax=ax2)
        plt.title('Consistency Across Conditions')
        plt.xticks(rotation=45)
        
        # 3. Topic Consistency Radar Chart
        ax3 = plt.subplot(4, 3, 3, projection='polar')
        
        # Select first candidate for radar chart
        candidate_1 = candidates[0]
        topic_scores = list(consistency_results[candidate_1]['topic_consistency'].values())
        
        angles = np.linspace(0, 2*np.pi, len(self.categories), endpoint=False)
        
        ax3.plot(angles, topic_scores, 'o-', linewidth=2, label=f'Candidate {candidate_1}')
        ax3.fill(angles, topic_scores, alpha=0.25)
        ax3.set_xticks(angles)
        ax3.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat for cat in self.categories], 
                           fontsize=8)
        ax3.set_ylim(0, 1)
        plt.title('Topic Consistency (Candidate 1)')
        
        # 4. Argument Susceptibility
        ax4 = plt.subplot(4, 3, 4)
        lib_susceptibility = [flexibility_results[c]['liberal_argument_susceptibility'] for c in candidates]
        cons_susceptibility = [flexibility_results[c]['conservative_argument_susceptibility'] for c in candidates]
        
        plt.scatter(lib_susceptibility, cons_susceptibility, s=100, alpha=0.7)
        for i, candidate in enumerate(candidates):
            plt.annotate(f'C{candidate}', (lib_susceptibility[i], cons_susceptibility[i]))
        
        plt.xlabel('Liberal Argument Susceptibility')
        plt.ylabel('Conservative Argument Susceptibility')
        plt.title('Argument Susceptibility Pattern')
        plt.grid(True, alpha=0.3)
        
        # 5. Role-playing Competence vs Consistency
        ax5 = plt.subplot(4, 3, 5)
        roleplay_scores = [composite_scores[c]['roleplay_competence'] for c in candidates]
        
        plt.scatter(consistency_scores, roleplay_scores, s=100, alpha=0.7)
        for i, candidate in enumerate(candidates):
            plt.annotate(f'C{candidate}', (consistency_scores[i], roleplay_scores[i]))
        
        plt.xlabel('Consistency Score')
        plt.ylabel('Role-playing Competence')
        plt.title('Consistency vs Role-playing Ability')
        plt.grid(True, alpha=0.3)
        
        # 6. Response Distribution by Condition
        ax6 = plt.subplot(4, 3, 6)
        condition_means = df.groupby('condition')['response'].mean()
        condition_means.plot(kind='bar', ax=ax6)
        plt.title('Average Response by Condition')
        plt.ylabel('Conservative Response Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Null Response Patterns
        ax7 = plt.subplot(4, 3, 7)
        null_rates = [composite_scores[c]['null_response_rate'] for c in candidates]
        plt.bar(range(len(candidates)), null_rates, alpha=0.7)
        plt.xlabel('Candidates')
        plt.ylabel('Null Response Rate')
        plt.title('Non-Response Patterns')
        plt.xticks(range(len(candidates)), [f'C{c}' for c in candidates])
        plt.grid(True, alpha=0.3)
        
        # 8. Category Response Distribution
        ax8 = plt.subplot(4, 3, 8)
        category_means = df.groupby('category')['response'].mean().sort_values()
        category_means.plot(kind='barh', ax=ax8)
        plt.title('Conservative Response Rate by Topic')
        plt.xlabel('Conservative Response Rate')
        plt.grid(True, alpha=0.3)
        
        # 9. Ideological Flexibility Spectrum
        ax9 = plt.subplot(4, 3, 9)
        overall_flexibility = [flexibility_results[c]['overall_argument_susceptibility'] for c in candidates]
        
        plt.scatter(range(len(candidates)), overall_flexibility, s=100, alpha=0.7)
        plt.plot(range(len(candidates)), overall_flexibility, '--', alpha=0.5)
        
        plt.xlabel('Candidates')
        plt.ylabel('Overall Flexibility Score')
        plt.title('Ideological Flexibility Spectrum')
        plt.xticks(range(len(candidates)), [f'C{c}' for c in candidates])
        plt.grid(True, alpha=0.3)
        
        # 10. Correlation Matrix of Key Metrics
        ax10 = plt.subplot(4, 3, 10)
        metrics_df = pd.DataFrame({
            'Depth': depth_scores,
            'Consistency': consistency_scores,
            'Flexibility': flexibility_scores,
            'Roleplay': roleplay_scores,
            'Null_Rate': null_rates
        })
        
        corr_matrix = metrics_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax10)
        plt.title('Metric Correlations')
        
        # 11. Individual Candidate Profile (Candidate 1)
        ax11 = plt.subplot(4, 3, 11)
        candidate_1_data = df[df['candidate_id'] == candidates[0]]
        condition_responses = candidate_1_data.groupby('condition')['response'].mean()
        
        condition_responses.plot(kind='bar', ax=ax11, color='skyblue', alpha=0.7)
        plt.title(f'Candidate {candidates[0]} Response Profile')
        plt.ylabel('Conservative Response Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 12. PCA Analysis
        ax12 = plt.subplot(4, 3, 12)
        
        # Prepare data for PCA
        pca_data = []
        for candidate in candidates:
            candidate_data = df[df['candidate_id'] == candidate]
            features = []
            for condition in self.conditions.values():
                cond_mean = candidate_data[candidate_data['condition'] == condition]['response'].mean()
                features.append(cond_mean if not np.isnan(cond_mean) else 0.5)
            pca_data.append(features)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100, alpha=0.7)
        for i, candidate in enumerate(candidates):
            plt.annotate(f'C{candidate}', (pca_result[i, 0], pca_result[i, 1]))
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA: Ideological Space')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("fig-1-updated.png")
        
        # return fig
    
    def generate_report(self, composite_scores, consistency_results, flexibility_results):
        """Generate a comprehensive analysis report"""
        print("=" * 80)
        print("POLITICAL IDEOLOGICAL DEPTH ANALYSIS REPORT")
        print("=" * 80)
        
        candidates = list(composite_scores.keys())
        
        print("\n1. OVERALL RANKINGS")
        print("-" * 40)
        
        # Rank by ideological depth
        depth_ranking = sorted(candidates, 
                             key=lambda x: composite_scores[x]['ideological_depth_score'], 
                             reverse=True)
        
        print("Ideological Depth Ranking:")
        for i, candidate in enumerate(depth_ranking, 1):
            score = composite_scores[candidate]['ideological_depth_score']
            print(f"  {i}. Candidate {candidate}: {score:.3f}")
        
        print("\n2. KEY METRICS SUMMARY")
        print("-" * 40)
        
        for candidate in candidates:
            print(f"\nCandidate {candidate}:")
            print(f"  • Ideological Depth Score: {composite_scores[candidate]['ideological_depth_score']:.3f}")
            print(f"  • Consistency Score: {composite_scores[candidate]['consistency_score']:.3f}")
            print(f"  • Flexibility Score: {composite_scores[candidate]['flexibility_score']:.3f}")
            print(f"  • Role-playing Competence: {composite_scores[candidate]['roleplay_competence']:.3f}")
            print(f"  • Null Response Rate: {composite_scores[candidate]['null_response_rate']:.3f}")
            print(f"  • Liberal Arg. Susceptibility: {flexibility_results[candidate]['liberal_argument_susceptibility']:.3f}")
            print(f"  • Conservative Arg. Susceptibility: {flexibility_results[candidate]['conservative_argument_susceptibility']:.3f}")
        
        print("\n3. INTERPRETIVE INSIGHTS")
        print("-" * 40)
        
        # Find most/least consistent
        most_consistent = max(candidates, key=lambda x: composite_scores[x]['consistency_score'])
        least_consistent = min(candidates, key=lambda x: composite_scores[x]['consistency_score'])
        
        print(f"Most Ideologically Consistent: Candidate {most_consistent}")
        print(f"Least Ideologically Consistent: Candidate {least_consistent}")
        
        # Find most/least flexible
        most_flexible = max(candidates, key=lambda x: composite_scores[x]['flexibility_score'])
        least_flexible = min(candidates, key=lambda x: composite_scores[x]['flexibility_score'])
        
        print(f"Most Ideologically Flexible: Candidate {most_flexible}")
        print(f"Least Ideologically Flexible: Candidate {least_flexible}")
        
        # Best role-player
        best_roleplayer = max(candidates, key=lambda x: composite_scores[x]['roleplay_competence'])
        print(f"Best Role-playing Ability: Candidate {best_roleplayer}")
        
        print("\n4. RECOMMENDATIONS FOR FURTHER ANALYSIS")
        print("-" * 40)
        print("• Examine topic-specific consistency patterns for policy expertise")
        print("• Investigate high null-response candidates for strategic non-commitment")
        print("• Analyze argument susceptibility patterns for persuasion vulnerability")
        print("• Consider contextual factors that might explain flexibility patterns")
        print("• Validate findings with additional question sets or different framings")

# Example usage
def main():
    # Initialize analyzer
    analyzer = IdeologicalDepthAnalyzer()
    
    # Load sample data (replace with your actual data loading)
    print("Loading sample data...")
    # df = analyzer.load_sample_data(n_candidates=2)
    df = pd.read_csv("analysis/all_labeled_votes.csv")
    # df = df.fillna(0.5) # slight change in variance
    # Calculate metrics
    print("Calculating consistency metrics...")
    consistency_results = analyzer.calculate_consistency_scores(df)
    print(consistency_results[1]['intra_condition_consistency'])
    print(consistency_results[0]['intra_condition_consistency'])
    print("Calculating flexibility metrics...")
    flexibility_results = analyzer.calculate_flexibility_metrics(df)
    
    print("Creating composite scores...")
    composite_scores = analyzer.create_composite_scores(consistency_results, flexibility_results)
    
    # Generate visualizations
    print("Generating visualizations...")
    analyzer.visualize_results(df, consistency_results, flexibility_results, composite_scores)
    
    # # Generate report
    # print("Generating analysis report...")
    # analyzer.generate_report(composite_scores, consistency_results, flexibility_results)
    
    # return df, consistency_results, flexibility_results, composite_scores

if __name__ == "__main__":
    # df, consistency_results, flexibility_results, composite_scores = main()
    main()