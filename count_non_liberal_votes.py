import json

gt_data_path = "data_for_STA/data/politically-liberal/test.json"


layer = 14
arg_type = "matching"
# model= "gemma-2-9b-it"
model= "llama-3.1-8b-it"
base_data_path = f"results/politically-liberal/backup/{model}_results_personality/main/base/eval_politically-liberal/{model}_base_{arg_type}_politically-liberal.result.json"
caa_data_path = "results/politically-liberal/backup/{model}_results_personality/main/caa/eval_politically-liberal/{model}_steer_politically-liberal_caa_{arg_type}_layer{layer}_multiplier={multiplier}.result.json"
sta_data_path = "results/politically-liberal/backup/{model}_results_personality/main/sta_act_and_fre_trim/eval_politically-liberal/trim0.35/{model}_steer_politically-liberal_{arg_type}_sae_caa_layer{layer}_act_and_fre_trim_trim0.35_mymulti={multiplier}.result.json"

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
