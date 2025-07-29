import matplotlib.pyplot as plt
import numpy as np

layer = 20
model= "gemma-2-9b-it"
plot_data = [
    [
        [(4, 46, 0)], 
        [(38, 12, 0), (34, 16, 0), (26, 24, 0), (25, 25, 0), (31, 19, 0), (32, 18, 0), (25, 25, 0), (22, 28, 0)], 
        [(42, 8, 0), (37, 13, 0), (28, 22, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (17, 33, 0), (8, 39, 3), (4, 35, 11)]
    ],
    [
        [(24, 26, 0)], 
        [(48, 2, 0), (47, 3, 0), (39, 11, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0)], 
        [(48, 2, 0), (48, 2, 0), (48, 2, 0), (44, 6, 0), (37, 13, 0), (33, 17, 0), (23, 27, 0), (17, 33, 0), (12, 38, 0), (7, 43, 0)]
    ],
]


# layer = 14
# model= "llama-3.1-8b-it"
# plot_data = [
#     [
#         [(3, 40, 7)], 
#         [(43, 4, 3), (29, 7, 14), (19, 10, 21), (36, 2, 12), (37, 8, 5), (38, 12, 0), (32, 18, 0), (30, 18, 0)], 
#         [(46, 3, 1), (41, 3, 6), (40, 5, 5), (44, 3, 3), (41, 8, 1), (38, 12, 0), (32, 16, 2), (29, 13, 8), (15, 4, 31), (0, 0, 50)]
#     ], 
#     [
#         [(14, 34, 2)], 
#         [(47, 3, 0), (36, 14, 0), (34, 16, 0), (30, 20, 0), (32, 18, 0), (29, 21, 0), (28, 22, 0), (16, 19, 3)], 
#         [(47, 3, 0), (46, 4, 0), (45, 5, 0), (39, 11, 0), (29, 21, 0), (19, 21, 10), (14, 10, 26), (0, 3, 47), (0, 0, 50), (0, 0, 50)]
#     ],
# ]


# Define x-values
x_vals_base = [0]
x_vals_caa = [-0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0]
x_vals_sta = [-0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0]

# Plot titles
titles = ['Base Model', 'CAA Steering', 'STA Steering']
x_values = [x_vals_base, x_vals_caa, x_vals_sta]

# Set up 1 row, 3 columns with different widths
fig = plt.figure(figsize=(20, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 4, 5])
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Colors and labels
colors = ['blue', 'red', 'green']
vote_types = ['Liberal', 'Conservative', 'Null']
conditions = ['Baseline', 'Argumentative']

# Bar width
bar_width = 0.13

# Plot each method: Base, CAA, STA
for i in range(3):
    ax = axs[i]
    x_pos = np.arange(len(x_values[i]))
    
    # For each condition (baseline, argumentative)
    for j in range(2):
        data = plot_data[j][i]
        
        # Extract vote counts and convert to percentages
        lib_data = [element[0] * (100/50) for element in data]
        cons_data = [element[1] * (100/50) for element in data]
        null_data = [element[2] * (100/50) for element in data]
        
        vote_data = [lib_data, cons_data, null_data]
        
        # Plot bars for each vote type
        for k, (votes, color, vote_type) in enumerate(zip(vote_data, colors, vote_types)):
            # Create spacing between baseline and argumentative groups
            group_offset = j * (3 * bar_width + 0.1)  # Add 0.1 spacing between groups
            bar_offset = k * bar_width
            position = x_pos + group_offset + bar_offset
            
            bars = ax.bar(position, votes, bar_width, 
                         color=color, alpha=0.7 if j == 0 else 1.0,
                         label=f'{conditions[j]} {vote_type}' if i == 0 else '',
                         edgecolor='black', linewidth=0.5)
    
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel('Multiplier', fontsize=12)
    ax.set_ylabel('% of votes', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_values[i]])
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    # Only show legend on first subplot
    if i == 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'comparison_layer_{layer}_{model}_grouped.png', dpi=300, bbox_inches='tight')