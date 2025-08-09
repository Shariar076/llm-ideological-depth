import matplotlib.pyplot as plt
import numpy as np

# layer = 20
# model= "gemma-2-9b-it"
# plot_data = [
#     [
#         [(4, 46, 0)], 
#         [(38, 12, 0), (34, 16, 0), (26, 24, 0), (25, 25, 0), (31, 19, 0), (32, 18, 0), (25, 25, 0), (22, 28, 0)], 
#         [(42, 8, 0), (37, 13, 0), (28, 22, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (17, 33, 0), (8, 39, 3), (4, 35, 11)]
#     ],
#     [
#         [(24, 26, 0)], 
#         [(48, 2, 0), (47, 3, 0), (39, 11, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0)], 
#         [(48, 2, 0), (48, 2, 0), (48, 2, 0), (44, 6, 0), (37, 13, 0), (33, 17, 0), (23, 27, 0), (17, 33, 0), (12, 38, 0), (7, 43, 0)]
#     ],
# ]
# plot_data = [[[(99, 7, 0), (12, 94, 0)], [(99, 7, 0), (82, 24, 0), (75, 31, 0), (60, 46, 0), (62, 44, 0), (68, 38, 0), (67, 39, 0), (52, 54, 0), (49, 57, 0), (40, 66, 0), (13, 93, 0)], [(99, 7, 0), (91, 15, 0), (78, 28, 0), (72, 34, 0), (65, 41, 0), (62, 44, 0), (59, 47, 0), (59, 47, 0), (51, 55, 0), (51, 55, 0), (51, 55, 0)]], [[(98, 8, 0), (36, 70, 0)], [(95, 11, 0), (99, 7, 0), (97, 9, 0), (80, 26, 0), (57, 49, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0)], [(95, 11, 0), (99, 7, 0), (97, 9, 0), (87, 19, 0), (79, 27, 0), (74, 32, 0), (70, 36, 0), (68, 38, 0), (71, 35, 0), (82, 24, 0), (69, 37, 0)]]]
# plot_data = [[[(99, 7, 0), (106, 0, 0)], [(99, 7, 0), (105, 1, 0), (106, 0, 0), (104, 2, 0), (98, 8, 0), (91, 15, 0), (91, 15, 0), (95, 11, 0), (95, 11, 0), (95, 11, 0), (92, 14, 0)], [(99, 7, 0), (102, 4, 0), (106, 0, 0), (106, 0, 0), (105, 1, 0), (102, 4, 0), (98, 8, 0), (95, 11, 0), (93, 13, 0), (93, 13, 0), (84, 22, 0)]], [[(98, 8, 0), (98, 8, 0)], [(95, 11, 0), (87, 19, 0), (79, 27, 0), (67, 39, 0), (56, 50, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (56, 50, 0), (58, 48, 0), (66, 40, 0)], [(95, 11, 0), (87, 19, 0), (75, 31, 0), (67, 39, 0), (62, 44, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (61, 45, 0)]]]

layer = 14
model= "llama-3.1-8b-it"
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

# plot_data = [[[(103, 3, 0), (6, 83, 17)], [(103, 3, 0), (88, 15, 3), (70, 14, 22), (39, 23, 44), (68, 10, 28), (79, 19, 8), (70, 36, 0), (66, 40, 0), (60, 46, 0), (59, 47, 0), (31, 15, 29)], [(101, 5, 0), (99, 6, 1), (95, 4, 7), (85, 6, 15), (89, 12, 5), (84, 20, 2), (74, 30, 2), (62, 39, 5), (55, 36, 15), (34, 13, 59), (0, 0, 106)]], [[(102, 4, 0), (40, 64, 2)], [(100, 6, 0), (100, 6, 0), (85, 21, 0), (73, 33, 0), (67, 39, 0), (58, 48, 0), (54, 52, 0), (59, 46, 0), (52, 47, 0), (38, 43, 1), (9, 11, 41)], [(102, 4, 0), (98, 8, 0), (102, 4, 0), (95, 11, 0), (72, 34, 0), (55, 51, 0), (50, 48, 8), (22, 28, 56), (2, 4, 100), (0, 0, 106), (0, 0, 106)]]]
plot_data = [[[(103, 3, 0), (104, 0, 2)], [(103, 3, 0), (106, 0, 0), (106, 0, 0), (103, 3, 0), (98, 8, 0), (99, 7, 0), (98, 8, 0), (96, 10, 0), (89, 17, 0), (83, 23, 0), (55, 51, 0)], [(101, 5, 0), (103, 3, 0), (105, 1, 0), (104, 2, 0), (106, 0, 0), (104, 2, 0), (105, 1, 0), (97, 9, 0), (82, 24, 0), (74, 32, 0), (26, 4, 0)]], [[(102, 4, 0), (106, 0, 0)], [(100, 6, 0), (104, 2, 0), (106, 0, 0), (98, 8, 0), (81, 25, 0), (64, 42, 0), (58, 48, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0), (55, 51, 0)], [(102, 4, 0), (104, 2, 0), (103, 3, 0), (105, 1, 0), (104, 2, 0), (97, 9, 0), (89, 17, 0), (84, 22, 0), (70, 36, 0), (54, 26, 0), (0, 0, 0)]]]

def plot_marked_lines():
    # Define x-values
    # being conservative
    # x_vals_base = ["role_none", "role_conservative"]
    # x_vals_caa = ["0.0", "-0.25", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-3.0", "-3.5", "-4.0", "-5.0"]
    # x_vals_sta = ["0.0", "-0.25", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-3.0", "-3.5", "-4.0", "-5.0"]
    # being liberal
    x_vals_base = ["role_none", "role_liberal"]
    x_vals_caa = ["0.0", "0.25", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0"]
    x_vals_sta = ["0.0", "0.25", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0"]
    
    # Flatten all y-values to get global min/max for consistent y-axis
    # all_values = sum([sum(setting, []) for setting in plot_data], [])
    y_min, y_max = 0, 100 #min(all_values), max(all_values)

    # Add a margin for better readability
    y_margin = (y_max - y_min) * 0.1
    y_min -= y_margin
    y_max += y_margin

    # Plot titles
    titles = ['Base Model', 'CAA Steering', 'STA Steering']

    # Set up 1 row, 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Colors for comparison
    markers = ['^', 's']
    linestyles = ['-', '--']
    labels = ['No Argument', 'Argumentative']

    # Plot each method: Base, CAA, STA
    for i in range(3):
        ax = axs[i]
        for j in range(2):  # 0 = baseline, 1 = argumentative
            data = plot_data[j][i]
            # Extract liberal, conservative, and null votes from tuples
            # mult  * (100/106) to get %
            lib_data =  [element[0] for element in data]
            cons_data = [element[1] for element in data]
            null_data = [element[2] for element in data]
            
            if i == 0:
                ax.plot(x_vals_base, lib_data, markers[j], label=f'{labels[j]} Lib Vote', color="blue", linestyle=linestyles[j])
                ax.plot(x_vals_base, cons_data, markers[j], label=f'{labels[j]} Cons Vote', color="red", linestyle=linestyles[j])
                ax.plot(x_vals_base, null_data, markers[j], label=f'{labels[j]} Null Vote', color="green", linestyle=linestyles[j])
            elif i == 1:
                ax.plot(x_vals_caa, lib_data, markers[j], label=f'{labels[j]} Lib Vote', color="blue", linestyle=linestyles[j])
                ax.plot(x_vals_caa, cons_data, markers[j], label=f'{labels[j]} Cons Vote', color="red", linestyle=linestyles[j])
                ax.plot(x_vals_caa, null_data, markers[j], label=f'{labels[j]} Null Vote', color="green", linestyle=linestyles[j])
            else:
                ax.plot(x_vals_sta, lib_data, markers[j], label=f'{labels[j]} Lib Vote', color="blue", linestyle=linestyles[j])
                ax.plot(x_vals_sta, cons_data, markers[j], label=f'{labels[j]} Cons Vote', color="red", linestyle=linestyles[j])
                ax.plot(x_vals_sta, null_data, markers[j], label=f'{labels[j]} Null Vote', color="green", linestyle=linestyles[j])
        
        ax.set_title(titles[i])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Multiplier')
        ax.set_ylabel('Number of non-liberal votes')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    # plt.savefig(f'analysis/comparison_layer_{layer}_{model}_being_conservative.png')
    plt.savefig(f'analysis/comparison_layer_{layer}_{model}_being_liberal.png')


def plot_grouped_bars():
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
    plt.savefig(f'analysis/comparison_layer_{layer}_{model}_grouped.png', dpi=300, bbox_inches='tight')
    
    
plot_marked_lines()
# plot_grouped_bars()