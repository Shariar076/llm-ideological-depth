import json

gt_data_path = "data_for_STA/data/politically-liberal/test_new.json"


layer = 14
# model= "gemma-2-9b-it"
model= "llama-3.1-8b-it"

# challeged while being conservative
# prompt_types = ["none", "conservative"]  
# arg_types = ["none", "matching"]
# multipliers = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0] # quality drop after , -5.0, -8, -10

# challeged while being liberal
prompt_types = ["none", "liberal"]  
arg_types = ["none", "matching"]
multipliers = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] 

base_data_path = "results/politically-liberal/{model}_results_personality_prompt/main/base/eval_politically-liberal/{model}_base_{prompt_type}_{arg_type}.result.json"
caa_data_path = "results/politically-liberal/{model}_results_personality_steering/main/caa/eval_politically-liberal/{model}_steer_politically-liberal_caa_{arg_type}_layer{layer}_multiplier={multiplier}.result.json"
sta_data_path = "results/politically-liberal/{model}_results_personality_steering/main/sta_act_and_fre_trim/eval_politically-liberal/trim0.35/{model}_steer_politically-liberal_{arg_type}_sae_caa_layer{layer}_act_and_fre_trim_trim0.35_mymulti={multiplier}.result.json"

gt_data = json.load(open(gt_data_path, "r"))

def count_mismatched_votes(data):
    lib_votes = 0
    con_votes = 0
    null_votes = 0
    assert len(data)==len(gt_data), "Data len don't match"
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
    # print("null_votes:", null_votes)
    return lib_votes, con_votes, null_votes

plot_data = [
    [
        [], 
        [], 
        []
    ], 
    [
        [], 
        [], 
        []
    ],
]

for idx, arg_type in enumerate(arg_types):
    vote_counts = []
    for prompt_type in prompt_types:
        base_data = json.load(open(base_data_path.format(arg_type=arg_type, prompt_type=prompt_type, model=model), "r"))
        vote_counts.append(count_mismatched_votes(base_data))
    plot_data[idx][0]=vote_counts    
    print(f"Role {prompt_types} Argument: {arg_type}", vote_counts)


for idx, arg_type in enumerate(arg_types):
    vote_counts = []
    for mul in multipliers: 
        caa_data = json.load(open(caa_data_path.format(arg_type=arg_type, multiplier=mul, layer=layer, model=model), "r"))
        vote_counts.append(count_mismatched_votes(caa_data))
    plot_data[idx][1]=vote_counts
    print(f"Steering CAA Argument: {arg_type}", vote_counts)

for idx, arg_type in enumerate(arg_types):
    vote_counts = []
    for mul in multipliers:
        sta_data = json.load(open(sta_data_path.format(arg_type=arg_type, multiplier=mul, layer=layer, model=model), "r"))
        vote_counts.append(count_mismatched_votes(sta_data))
    plot_data[idx][2]=vote_counts
    print(f"Steering STA Argument: {arg_type}", vote_counts)

print(plot_data)
