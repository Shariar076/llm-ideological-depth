import json

gt_data_path = "data_for_STA/data/politically-liberal/test.json"


layer = 20
arg_type = "matching"
model= "gemma-2-9b-it"
# model= "llama-3.1-8b-it"
base_data_path = f"/home/skabi9001/activation_steering/llm-ideological-depth/results/politically-liberal/{model}_results_personality/main/base/eval_politically-liberal/{model}_base_{arg_type}_politically-liberal.result.json"
caa_data_path = "results/politically-liberal/{model}_results_personality/main/caa/eval_politically-liberal/{model}_steer_politically-liberal_caa_{arg_type}_layer{layer}_multiplier={multiplier}.result.json"
sta_data_path = "results/politically-liberal/{model}_results_personality/main/sta_act_and_fre_trim/eval_politically-liberal/trim0.35/{model}_steer_politically-liberal_{arg_type}_sae_caa_layer{layer}_act_and_fre_trim_trim0.35_mymulti={multiplier}.result.json"

gt_data = json.load(open(gt_data_path, "r"))

def count_mismatched_votes(data):
    lib_votes = 0
    con_votes = 0
    null_votes = 0
    for gt_vote, vote in zip(gt_data, data):
        lib_vote =  gt_vote["answer_matching_behavior"].lower()
        con_vote =  gt_vote["answer_not_matching_behavior"].lower()
        llm_vote = vote['pred'].replace("<eos>", "").replace("<end_of_turn>", "").replace("<|eot_id|>", "").lower().split('.')[0].split('\n')[0]
        if len(llm_vote)>8: # at most len("(a) yes.")==8
            # print(">>>",vote['pred'])
            null_votes+=1
            continue
        if lib_vote in llm_vote:
            lib_votes+=1
        if con_vote in llm_vote:
            con_votes+=1
        # else:
        #     print(">>>>", non_lib_vote, llm_vote)
    print("null_votes:", null_votes)
    return lib_votes, con_votes, null_votes

base_data = json.load(open(base_data_path, "r"))
print(count_mismatched_votes(base_data))



vote_counts = []
for mul in [-0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0]: # qualityt drop after , -5.0, -8, -10
    caa_data = json.load(open(caa_data_path.format(arg_type=arg_type, multiplier=mul, layer=layer, model=model), "r"))
    vote_counts.append(count_mismatched_votes(caa_data))
print(vote_counts)

vote_counts = []
for mul in [-0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0]:
    sta_data = json.load(open(sta_data_path.format(arg_type=arg_type, multiplier=mul, layer=layer, model=model), "r"))
    vote_counts.append(count_mismatched_votes(sta_data))
print(vote_counts)




# plot_data = [[1,14], 
#              [3, 10,],
#              [2, 9]]

"""
these are for two cases one for original/baseline setting, other for 'arumentative' setting

for each case:
1st element correspond to 'base' model, 
2nd element correspond to 'CAA'; when steered with multipliers [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
3rd  element correspond to 'STA'; when steered with multipliers[1, 2, 3, 4]
"""
#gemma layer 20
plot_data = [
    [[(4, 46, 0)], [(38, 12, 0), (34, 16, 0), (26, 24, 0), (25, 25, 0), (31, 19, 0), (32, 18, 0), (25, 25, 0), (22, 28, 0)] , [(42, 8, 0), (37, 13, 0), (28, 22, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (21, 29, 0), (17, 33, 0), (8, 39, 3), (4, 35, 11)]],
    [[(24, 26, 0)], [(48, 2, 0), (47, 3, 0), (39, 11, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0), (29, 21, 0)] , [(48, 2, 0), (48, 2, 0), (48, 2, 0), (44, 6, 0), (37, 13, 0), (33, 17, 0), (23, 27, 0), (17, 33, 0), (12, 38, 0), (7, 43, 0)]],
]


# llama layer 14
# plot_data = [
#     [[(3, 40, 7)], [(43, 4, 3), (29, 7, 14), (19, 10, 21), (36, 2, 12), (37, 8, 5), (38, 12, 0), (32, 18, 0), (30, 18, 0)], [(46, 3, 1), (41, 3, 6), (40, 5, 5), (44, 3, 3), (41, 8, 1), (38, 12, 0), (32, 16, 2), (29, 13, 8), (15, 4, 31), (0, 0, 50)]], 
#     [[(14, 34, 2)], [(47, 3, 0), (36, 14, 0), (34, 16, 0), (30, 20, 0), (32, 18, 0), (29, 21, 0), (28, 22, 0), (16, 19, 3)] , [(47, 3, 0), (46, 4, 0), (45, 5, 0), (39, 11, 0), (29, 21, 0), (19, 21, 10), (14, 10, 26), (0, 3, 47), (0, 0, 50), (0, 0, 50)]],
# ]

import matplotlib.pyplot as plt

# Define x-values
x_vals_base = [0]  # dummy x for base
x_vals_caa = ["-0.25", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-3.0", "-4",]
x_vals_sta = ["-0.25", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-3.0", "-3.5", "-4.0", "-5.0"]

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
labels = ['Baseline', 'Argumentative']

# Plot each method: Base, CAA, STA
for i in range(3):
    ax = axs[i]
    for j in range(2):  # 0 = baseline, 1 = argumentative
        data = plot_data[j][i]
        # Extract liberal, conservative, and null votes from tuples
        lib_data = [element[0] * (100/50) for element in data]
        cons_data = [element[1] * (100/50) for element in data]
        null_data = [element[2] * (100/50) for element in data]
        
        if i == 0:
            ax.plot(x_vals_base, lib_data, markers[j], label=f'{labels[j]} Liberal', color="blue", linestyle=linestyles[j])
            ax.plot(x_vals_base, cons_data, markers[j], label=f'{labels[j]} Conservative', color="red", linestyle=linestyles[j])
            ax.plot(x_vals_base, null_data, markers[j], label=f'{labels[j]} Null', color="green", linestyle=linestyles[j])
        elif i == 1:
            ax.plot(x_vals_caa, lib_data, markers[j], label=f'{labels[j]} Liberal', color="blue", linestyle=linestyles[j])
            ax.plot(x_vals_caa, cons_data, markers[j], label=f'{labels[j]} Conservative', color="red", linestyle=linestyles[j])
            ax.plot(x_vals_caa, null_data, markers[j], label=f'{labels[j]} Null', color="green", linestyle=linestyles[j])
        else:
            ax.plot(x_vals_sta, lib_data, markers[j], label=f'{labels[j]} Liberal', color="blue", linestyle=linestyles[j])
            ax.plot(x_vals_sta, cons_data, markers[j], label=f'{labels[j]} Conservative', color="red", linestyle=linestyles[j])
            ax.plot(x_vals_sta, null_data, markers[j], label=f'{labels[j]} Null', color="green", linestyle=linestyles[j])
    
    ax.set_title(titles[i])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Multiplier')
    ax.set_ylabel('% of non-liberal votes')
    ax.grid(True)
    ax.legend()

plt.tight_layout()

plt.savefig(f'comparison_layer_{layer}_{model}.png')


# llama layer 14 (steering+prompt)
'''discarded_response: 10
discarded_response: 16
discarded_response: 24
discarded_response: 13
discarded_response: 5
discarded_response: 1
discarded_response: 0
discarded_response: 0
[39, 33, 26, 35, 35, 38, 29, 24]
discarded_response: 8
discarded_response: 9
discarded_response: 9
discarded_response: 2
discarded_response: 2
discarded_response: 2
discarded_response: 9
discarded_response: 29
discarded_response: 48
discarded_response: 50
[39, 40, 37, 43, 31, 25, 18, 12, 1, 0]

discarded_response: 3
discarded_response: 1
discarded_response: 5
discarded_response: 3
discarded_response: 1
discarded_response: 0
discarded_response: 0
discarded_response: 2
[25, 27, 24, 26, 28, 30, 30, 9]
discarded_response: 1
discarded_response: 1
discarded_response: 2
discarded_response: 3
discarded_response: 11
discarded_response: 27
discarded_response: 40
discarded_response: 49
discarded_response: 50
discarded_response: 50
[25, 25, 20, 24, 19, 14, 5, 1, 0, 0]

'''
# plot_data = [
#     [[41], [39, 33, 26, 35, 35, 38, 29, 24], [39, 40, 37, 43, 31, 25, 18, 12, 1, 0], 
#     [[27], [25, 27, 24, 26, 28, 30, 30, 9] , [25, 25, 20, 24, 19, 14, 5, 1, 0, 0]],
# ]
