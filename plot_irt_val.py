import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

'''
# Chart data; don't need
chart_data = {
        "chart_1": {
            "title": "MIRT Correlation with DW NOMINATE over Repeated Trials",
        "x_axis": "Trials",
        "y_axis": "Abs. values",
        "data": {
        "Corr-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.947952, 0.948092, -0.981536, -0.948178, -0.915431, 0.227386, 0.914912, -0.057003, -0.043789, -0.046273]
        },
        "Corr-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [-0.322302, 0.105965, 0.105418, 0.016801, -0.153380, -0.351018, -0.801913, 0.356296, 0.802256, -0.104075]
        },
        "P_val-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.021540, 0.000000, 0.569308, 0.662105, 0.644211]
        },
        "P_val-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000955, 0.289147, 0.291660, 0.866899, 0.123781, 0.000298, 0.000000, 0.000237, 0.000000, 0.297884]
        }
        },
        "subplot_2": {
        "title": "MIRT Correlation with R-IDEAL over Repeated Trials",
        "data": {
            "Corr-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [-0.951060, -0.951029, 0.999404, 0.951133, 0.953370, -0.215182, -0.953263, -0.034640, -0.047831, -0.044577]
            },
            "Corr-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.541979, -0.242562, 0.120993, 0.000000, 0.000000, 0.177226, 0.976999, -0.639160, -0.977476, -0.122415]
            },
            "P_val-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.000000, 0.000000, -0.065185, 0.417563, 0.034287, 0.000000, 0.736237, 0.641757, 0.664611]
            },
            "P_val-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.016672, 0.237786, 0.525850, 0.000021, 0.082454, 0.000000, 0.000000, 0.000000, 0.232276]
            }
        }
        }
    },
    # "chart_2": {
    #     "title": "MIRT Correlation with DW NOMINATE over Repeated Trials",
    #     "x_axis": "Trials",
    #     "y_axis": "Abs. values",
    #     "data": {
    #     "Corr-Dim1": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.98, 0.99, 0.97, 0.98, 0.99, 0.99, 0.99, 0.90, 0.97, 0.75]
    #     },
    #     "Corr-DIM2": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.32, 0.05, 0.10, 0.08, 0.05, 0.02, 0.05, 0.78, 0.12, 0.22]
    #     },
    #     "P_val-Dim1": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    #     },
    #     "P_val-DIM2": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.40, 0.65, 0.35, 0.45, 0.18, 0.00, 0.00, 0.00, 0.25, 0.08]
    #     }
    #     },
    #     "subplot_2": {
    #     "title": "MIRT Correlation with R-IDEAL over Repeated Trials",
    #     "data": {
    #         "Corr-Dim1": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.95, 0.98, 0.99, 0.85]
    #         },
    #         "Corr-DIM2": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.42, 0.05, 0.18, 0.20, 0.78, 0.99, 0.42, 0.88, 0.18, 0.08]
    #         },
    #         "P_val-Dim1": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    #         },
    #         "P_val-DIM2": {
    #         "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "y": [0.72, 0.18, 0.22, 0.18, 0.00, 0.00, 0.18, 0.88, 0.12, 0.05]
    #         }
    #     }
    #     }
    # },
    "chart_3": {
        "title": "MIRT Correlation with DW NOMINATE over Repeated Trials",
        "x_axis": "Trials",
        "y_axis": "Abs. values",
        "data": {
        "Corr-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.850735, 0.915586, -0.802023, 0.743871, 0.903525, 0.974032, 0.968500, 0.971474, 0.974002, 0.941240]
        },
        "Corr-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, -0.817493, -0.028680, -0.147571, -0.364829, -0.520493, -0.361506, -0.778028, -0.520367, -0.745437]
        },
        "P_val-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [-0.528698, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        },
        "P_val-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.000000, 0.774768, 0.138836, 0.000163, 0.000000, 0.000189, 0.000000, 0.000000, 0.000000]
        }
        },
        "subplot_2": {
        "title": "MIRT Correlation with R-IDEAL over Repeated Trials",
        "data": {
            "Corr-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.836844, 0.948997, -0.857422, 0.706959, 0.936991, 0.984157, 0.968196, 0.983078, 0.984040, 0.939067]
            },
            "Corr-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [-0.660258, -0.953032, -0.061536, -0.177381, -0.414567, -0.628234, -0.429290, -0.937413, -0.628411, -0.953862]
            },
            "P_val-Dim1": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
            },
            "P_val-DIM2": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0.000000, 0.000000, 0.547219, 0.080579, 0.000022, 0.000000, 0.000010, 0.000000, 0.000000, 0.000000]
            }
        }
        }
    }
}
'''

# --- 1. Input your data here ---
# The data is structured as a list of dictionaries for clarity.
data = [
    # 3 Reference Points (Top Group)
    {'model': '3 References', 'target': 'DW Nominate', 'dim': 'Dim 1', 'mean': 0.9095363945159471, 'std': 0.17344316237925397},
    {'model': '3 References', 'target': 'DW Nominate', 'dim': 'Dim 2', 'mean': 0.6462287176598804, 'std': 0.17422965482293842},
    {'model': '3 References', 'target': 'R-IDEAL', 'dim': 'Dim 1', 'mean': 0.9095363945159471, 'std': 0.1610419335161174},
    {'model': '3 References', 'target': 'R-IDEAL', 'dim': 'Dim 2', 'mean': 0.6462287176598804, 'std': 0.23284477650783764},
    
    # 2 Reference Points (Middle Group)
    {'model': '2 References', 'target': 'DW Nominate', 'dim': 'Dim 1', 'mean': 0.9459200965569885, 'std': 0.06594310359577675},
    {'model': '2 References', 'target': 'DW Nominate', 'dim': 'Dim 2', 'mean': 0.3285690720322404, 'std': 0.2818463867981155},
    {'model': '2 References', 'target': 'R-IDEAL', 'dim': 'Dim 1', 'mean': 0.9459200965569885, 'std': 0.07941234411276368},
    {'model': '2 References', 'target': 'R-IDEAL', 'dim': 'Dim 2', 'mean': 0.3285690720322404, 'std': 0.3202766234269498},

    # No Reference Points (Bottom Group)
    {'model': 'No References', 'target': 'DW Nominate', 'dim': 'Dim 1', 'mean': 0.6030553363570369, 'std': 0.41919933656329406},
    {'model': 'No References', 'target': 'DW Nominate', 'dim': 'Dim 2', 'mean': 0.31194250513571953, 'std': 0.26918738057625446},
    {'model': 'No References', 'target': 'R-IDEAL', 'dim': 'Dim 1', 'mean': 0.6030553363570369, 'std': 0.43116351144632914},
    {'model': 'No References', 'target': 'R-IDEAL', 'dim': 'Dim 2', 'mean': 0.31194250513571953, 'std': 0.3282616443681644},
]

# --- 2. Set up plot styling for a research paper ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(8, 10))

# --- 3. Define colors and layout ---
colors = {'Dim 1': '#0072B2', 'Dim 2': '#31a354'} # Colorblind-friendly blue and orange
group_spacing = 1.5  # Space between major groups (0, 2, 3 refs)
item_spacing = 0.5   # Space between items within a group

# Calculate y-positions for each point
y_pos = []
current_y = 0
for i, entry in enumerate(data):
    y_pos.append(current_y)
    current_y -= 1
    # Add extra space after each sub-group of 2 (DW vs IDEAL)
    if (i + 1) % 2 == 0:
        current_y -= item_spacing
    # Add a larger space after each main group of 4
    if (i + 1) % 4 == 0:
        current_y -= group_spacing
        
# --- 4. Plot the data ---
y_labels = []
for i, entry in enumerate(data):
    mean = entry['mean']
    std = entry['std']
    dim = entry['dim']
    color = colors[dim]

    # Plot the shaded region for standard deviation (mean ± std)
    # Using a thick, semi-transparent horizontal line to mimic the shaded effect
    ax.hlines(y=y_pos[i], xmin=mean - std, xmax=mean + std, color=color, alpha=0.25, linewidth=12, zorder=1)
    
    # Plot the mean as a solid dot
    ax.plot(mean, y_pos[i], 'o', color=color, markersize=8, zorder=2)
    
    # Store the label for the y-axis
    y_labels.append(f"vs. {entry['target']}")


# --- 5. Add group labels and refine aesthetics ---
# Add labels for the main identification strategies
ax.text(1.12, y_pos[1], '3 References', fontsize=14, ha='left', va='center', fontweight='bold')
ax.axhline(y_pos[3]-(group_spacing), color='gray', linestyle='--', linewidth=0.8, zorder=0)
ax.text(1.12, y_pos[5], '2 References', fontsize=14, ha='left', va='center', fontweight='bold')
ax.axhline(y_pos[7]-(group_spacing), color='gray', linestyle='--', linewidth=0.8, zorder=0)
ax.text(1.12, y_pos[9], 'No References', fontsize=14, ha='left', va='center', fontweight='bold')

# Customize the axes and grid
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels)
ax.tick_params(axis='y', length=0) # Hide y-axis ticks

ax.set_xlabel('Correlation with External Scores')
# ax.set_title('Model Performance by Identification Strategy', pad=20)
ax.set_xlim(-0.05, 1.1) # Set x-axis limits

# Add a vertical line at 0 for reference (no correlation)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add a light horizontal grid
ax.grid(axis='x', linestyle=':', color='gray', alpha=0.6)

# --- 6. Create a custom legend ---
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='Dimension 1',
                  markerfacecolor=colors['Dim 1'], markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='Dimension 2',
                  markerfacecolor=colors['Dim 2'], markersize=10)
]
ax.legend(handles=legend_elements, loc='upper center', frameon=False, ncol=1)

fig.tight_layout()

# --- 7. Save the figure ---
plt.savefig('mirt_correlation_comparison.pdf', bbox_inches='tight')
plt.savefig('mirt_correlation_comparison.png', dpi=300, bbox_inches='tight')

print("Plot saved as 'mirt_correlation_comparison.pdf' and 'mirt_correlation_comparison.png'")