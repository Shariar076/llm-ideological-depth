import os
import json
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def plot_ideal_points(ideal_points, file_name):
    print("Creating plot...")

    # Set up colors
    party_colors = {'D': 'blue', 'R': 'red', 'I': 'green', 'U': 'purple'}
    colors = [party_colors[party] for party in member_info_df['party']]

    # Create plot
    # plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (5,5)})

    # Scatter plot of ideal points (IRT person parameters)
    plt.scatter(ideal_points[:, 0], ideal_points[:, 1], 
                c=colors, s=50, alpha=0.7)

    # Add labels for each point
    for i, row in member_info_df.iterrows():
        # Format name like R output: "LASTNAME (PARTY STATE)"
        name_parts = row['bioname'].split(', ')
        if len(name_parts) >= 2:
            lastname = name_parts[0]
        else:
            lastname = row['bioname']
        
        # label = f"{lastname} ({row['party']} {row['state_abbrev']})"
        # label = f"{lastname} ({row['party']})"
        label = "" #row['bioname']
        
        plt.annotate(label, 
                    (ideal_points[i, 0], ideal_points[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, rotation=-15, 
                    ha='left', va='center', rotation_mode='anchor')

    # Formatting
    plt.xlabel('Ideology Dimension 1')  # First IRT dimension (usually liberal-conservative)
    plt.ylabel('Ideology Dimension 2')  # Second IRT dimension
    plt.title('2D Ideal Points by Party')

    # Legend
    legend_elements = [
        plt.scatter([], [], c='blue', s=50, label='Democrat'),
        plt.scatter([], [], c='red', s=50, label='Republican'),
        plt.scatter([], [], c='green', s=50, label='Independent'),
        # plt.scatter([], [], c='purple', s=100, label='LLM')
    ]
    plt.legend(handles=legend_elements)#, loc='upper left'

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"Saving figure at temp/ideal_points_{file_name}.png")
    plt.savefig(f"temp/ideal_points_{file_name}.png")
    # plt.show()


if __name__=="__main__":
    members = pd.read_csv('irt_files/S109_members.csv')
    votes = pd.read_csv('irt_files/S109_votes.csv')
    rollcalls = pd.read_csv('irt_files/S109_rollcalls.csv')

    print(f"Loaded {len(votes)} votes, {len(members)} members, {len(rollcalls)} roll calls")

    votes_senate = votes.copy()
    members_senate = members.copy()
    rollcalls_senate = rollcalls.copy()

    # Create mapping dictionaries
    member_to_idx = {icpsr: idx for idx, icpsr in enumerate(sorted(members_senate['icpsr'].unique()))}
    rollcall_to_idx = {roll: idx for idx, roll in enumerate(sorted(rollcalls_senate['rollnumber'].unique()))}

    # Create reverse mappings
    idx_to_member = {idx: icpsr for icpsr, idx in member_to_idx.items()}
    idx_to_rollcall = {idx: roll for roll, idx in rollcall_to_idx.items()}

    n_members = len(member_to_idx)
    n_rollcalls = len(rollcall_to_idx)

    print(f"Matrix dimensions: {n_members} members x {n_rollcalls} roll calls")

    vote_matrix = np.full((n_members, n_rollcalls), np.nan)

    # Fill in the vote matrix
    # Cast codes: 1,2,3 = Yea, 4,5,6 = Nay, 7,8,9 = Not voting/Present
    for _, row in votes_senate.iterrows():
        if row['icpsr'] in member_to_idx and row['rollnumber'] in rollcall_to_idx:
            member_idx = member_to_idx[row['icpsr']]
            roll_idx = rollcall_to_idx[row['rollnumber']]
            
            # Convert cast codes to binary votes
            if row[f'cast_code'] in [1, 2, 3]:  # Yea votes
                vote_matrix[member_idx, roll_idx] = 1
            elif row[f'cast_code'] in [4, 5, 6]:  # Nay votes
                vote_matrix[member_idx, roll_idx] = 0

    # Create member info dataframe with proper ordering
    member_info = []
    for idx in range(n_members):
        icpsr = idx_to_member[idx]
        member_row = members_senate[members_senate['icpsr'] == icpsr].iloc[0]
        member_info.append({
            'idx': idx,
            'icpsr': icpsr,
            'bioname': member_row['bioname'],
            'state_abbrev': member_row['state_abbrev'],
            'party_code': member_row['party_code'],
            'nominate_dim_1': member_row['nominate_dim1'],
            'nominate_dim_2': member_row['nominate_dim2']
        })

    member_info_df = pd.DataFrame(member_info)
    # print(member_info_df)

    # Convert party codes to letters (100=D, 200=R, others=I)
    party_map = {100: 'D', 200: 'R', 328: 'I'}
    member_info_df['party'] = member_info_df['party_code'].map(party_map).fillna('U')
    member_info_df['members_abbv'] = member_info_df.apply(lambda row: f"{row['bioname'].split(', ')[0]} ({row['party']} {row['state_abbrev']})", axis=1)
    
    # Remove roll calls with too few votes or unanimous votes
    valid_rollcalls = []
    valid_vote_matrix = []

    for roll_idx in range(n_rollcalls):
        votes_col = vote_matrix[:, roll_idx]
        valid_votes = votes_col[~np.isnan(votes_col)]
        
        if len(valid_votes) >= 2:  # At least 10 votes
            yea_count = np.sum(valid_votes == 1)
            nay_count = np.sum(valid_votes == 0)
            
            # Skip unanimous votes (no discrimination power, like items everyone gets right/wrong)
            if yea_count > 0 and nay_count > 0:
                valid_rollcalls.append(roll_idx)
                valid_vote_matrix.append(votes_col)

    vote_matrix_clean = np.column_stack(valid_vote_matrix)

    print(f"After filtering: {vote_matrix_clean.shape[0]} members x {vote_matrix_clean.shape[1]} roll calls")

    vote_matrix = vote_matrix_clean

    n_dims=2
    n_members, n_rollcalls = vote_matrix.shape
    # Create masks for observed votes
    observed_mask = ~np.isnan(vote_matrix)
    # Convert to binary format for observed votes only
    vote_obs = vote_matrix[observed_mask].astype(int)
    # Get indices for observed votes
    member_idx, rollcall_idx = np.where(observed_mask)
    print(f"Fitting {n_dims}D ideal point model...")
    print("n_members:", n_members)
    print("n_rollcalls:", n_rollcalls)
    print("n_observations:", len(vote_obs))

    stan_data = {
        'J': n_members,
        'K': n_rollcalls, 
        'N': len(vote_obs),
        'D': n_dims,
        'jj': member_idx+1, # starting idx 1
        'kk': rollcall_idx+1, # starting idx 1
        'y': vote_obs,
    }

    import arviz as az
    from cmdstanpy import CmdStanModel
    from scipy.stats import pearsonr

    stan_file = os.path.join('irt_files/MIRT.stan')
    model = CmdStanModel(stan_file=stan_file)


    y1 = []
    y2 = []
    y3 = []
    y4 = []

    y5 = []
    y6 = []
    y7 = []
    y8 = []


    n_samples=2000
    n_tune=2000
    for idx in range(10):
        fit = model.sample(
            data=stan_data, chains=4, 
            iter_sampling=n_samples, 
            iter_warmup=n_tune,
            # show_console=True
        )
        inference_data = az.from_cmdstanpy(fit)
        ideal_points_samples = inference_data.posterior['theta'].values
        # Average across chains and samples to get point estimates
        ideal_points = np.mean(ideal_points_samples, axis=(0, 1))
        
        plot_ideal_points(ideal_points, f"{idx}")
        
        merged_df_1 = member_info_df.join(pd.DataFrame(ideal_points, columns=['ideal_dim_1', 'ideal_dim_2']))
        # print(merged_df)
        # Calculate correlations
        corr1_dim1, p_value1_dim1 = pearsonr(merged_df_1['nominate_dim_1'], merged_df_1['ideal_dim_1'])
        corr1_dim2, p_value1_dim2 = pearsonr(merged_df_1['nominate_dim_2'], merged_df_1['ideal_dim_2'])
        print(f"{idx}. IRT-Stan v DW Nominate Correlation:")
        print("Dimension 1:")
        print(f"Correlation: {corr1_dim1:.6f}")
        print(f"P-value: {p_value1_dim1:.6f}")

        print("Dimension 2:")
        print(f"Correlation: {corr1_dim2:.6f}")
        print(f"P-value: {p_value1_dim2:.6f}\n\n")
        
        R_manual_df = pd.read_csv("irt_files/ideal_points_R_preloaded_files.csv")
        R_manual_df.columns =['members_abbv','ideal_dim_1_true', 'ideal_dim_2_true']
        merged_df_2 = merged_df_1.join(R_manual_df.set_index('members_abbv'), on='members_abbv', how='inner')
        corr2_dim1, p_value2_dim1 = pearsonr(merged_df_2['ideal_dim_1_true'], merged_df_2['ideal_dim_1'])
        corr2_dim2, p_value2_dim2 = pearsonr(merged_df_2['ideal_dim_2_true'], merged_df_2['ideal_dim_2'])
        print(f"{idx}. IRT-Stan v IDEAL-R Correlation:")
        print("Dimension 1:")
        print(f"Correlation: {corr2_dim1:.6f}")
        print(f"P-value: {p_value2_dim1:.6f}")

        print("Dimension 2:")
        print(f"Correlation: {corr2_dim2:.6f}")
        print(f"P-value: {p_value2_dim2:.6f}\n\n")
        
        
        y1.append(abs(corr1_dim1))
        y3.append(abs(p_value1_dim1))
        y2.append(abs(corr1_dim2))
        y4.append(abs(p_value1_dim2))
        
        y5.append(abs(corr2_dim1))
        y7.append(abs(p_value2_dim1))
        y6.append(abs(corr2_dim2))
        y8.append(abs(p_value2_dim2))
        
    # y1 = [0.981372, -0.051767, 0.947342, 0.981422, 0.065892, 0.487796, 0.976800, 0.947819, 0.959838, 0.502015]
    # y3 = [0.000000, 0.605347, 0.000000, 0.000000, 0.510547, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    # y2 = [0.006376, 0.026858, 0.105955, 0.103341, 0.102624, 0.119664, 0.102663, 0.103987, 0.800938, 0.105391]
    # y4 = [0.949287, 0.788730, 0.289196, 0.301325, 0.304709, 0.230933, 0.304524, 0.298295, 0.000000, 0.291784]

    print("DW Nominate Correlation:")
    print("Dim 1:", np.mean(y1), "±", np.std(y1))
    print("Dim 2:", np.mean(y2), "±", np.std(y2))
    print("R-IDEAL Correlation:")
    print("Dim 1:", np.mean(y1), "±", np.std(y5))
    print("Dim 2:", np.mean(y2), "±", np.std(y6))

    x = np.linspace(1, 10, 10)

    # Create the plot
    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))


    ax1.plot(x, y1, label='Corr-Dim1', color='blue', linewidth=2)
    ax1.plot(x, y2, label='Corr-DIM2', color='red', linewidth=2)
    ax1.plot(x, y3, label='P_val-Dim1', color='blue', linewidth=2, linestyle="--")
    ax1.plot(x, y4, label='P_val-DIM2', color='red', linewidth=2, linestyle="--")
    ax1.set_xlabel('Trials')
    ax1.set_ylabel('Abs. values')
    ax1.set_title('MIRT Correlation with DW NOMINATE over Repeated Trials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, y5, label='Corr-Dim1', color='blue', linewidth=2)
    ax2.plot(x, y6, label='Corr-DIM2', color='red', linewidth=2)
    ax2.plot(x, y7, label='P_val-Dim1', color='blue', linewidth=2, linestyle="--")
    ax2.plot(x, y8, label='P_val-DIM2', color='red', linewidth=2, linestyle="--")
    ax2.set_xlabel('Trials')
    ax2.set_ylabel('Abs. values')
    ax2.set_title('MIRT Correlation with R-IDEAL over Repeated Trials')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"MIRT-Correlations-n_tune={n_tune}-n_samples={n_samples}_1.png")
    # plt.savefig(f"test.png")
    
    
'''
DW Nominate Correlation:
Dim 1: 0.6969731899984482 ± 0.42659126464723085
Dim 2: 0.36614543786857495 ± 0.30228568624534913
R-IDEAL Correlation:
Dim 1: 0.6969731899984482 ± 0.43796922662006876
Dim 2: 0.36614543786857495 ± 0.3752729919721361
'''