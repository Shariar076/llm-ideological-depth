import json
import pandas as pd


gt_data_path = "data_for_STA/data/politically-liberal/test_new.json"
gt_data = json.load(open(gt_data_path, "r"))

categorized_statements_path = "data_for_STA/data/politically-liberal/categorized_statements.json"
categorized_statements = json.load(open(categorized_statements_path, "r"))

def count_votes(data):
    lib_votes = 0
    con_votes = 0
    null_votes = 0
    for gt_vote, vote in zip(gt_data, data):
        lib_vote =  gt_vote["answer_matching_behavior"].lower()
        con_vote =  gt_vote["answer_not_matching_behavior"].lower()
        llm_vote = vote['pred'].replace("<eos>", "").replace("<end_of_turn>", "").replace("<|eot_id|>", "").lower().split('.')[0].split('\n')[0]
        # if len(llm_vote)>8: # at most len("(a) yes.")==8
        #     # print(">>>",vote['pred'])
        #     null_votes+=1
        #     continue
        if lib_vote in llm_vote and con_vote not in llm_vote:
            lib_votes+=1
        elif con_vote in llm_vote and lib_vote not in llm_vote:
            con_votes+=1
        else:
            null_votes+=1
        # else:
        #     print(">>>>", non_lib_vote, llm_vote)
    # print("null_votes:", null_votes)
    return lib_votes, con_votes, null_votes

def get_vote(vote):
    llm_vote = vote['pred'].replace("<eos>", "").replace("<end_of_turn>", "").replace("<|eot_id|>", "").lower().split('.')[0].split('\n')[0]
    if len(llm_vote)>8: # at most len("(a) yes.")==8
        # print(">>>",vote['pred'])
        return None
    if 'a' in llm_vote:
        return 1
    if 'b' in llm_vote:
        return 0

def get_vote_label(vote, q_idx):
    lib_vote =  gt_data[q_idx]["answer_matching_behavior"].lower()
    con_vote =  gt_data[q_idx]["answer_not_matching_behavior"].lower()
    llm_vote = vote['pred'].replace("<eos>", "").replace("<end_of_turn>", "").replace("<|eot_id|>", "").lower().split('.')[0].split('\n')[0]
    if len(llm_vote)>8: # at most len("(a) yes.")==8
        # print(">>>",vote['pred'])
        return None
    if lib_vote in llm_vote:
        return 1
    if con_vote in llm_vote:
        return 0

def get_ques_topic(q_idx):
    statement = gt_data[q_idx]['statement']
    for topic, statements in categorized_statements.items():
        if statement in statements:
            return topic
    return None

# model="gemma-2-9b-it"
# model="llama-3.1-8b-it"
# used_prompts = ["none", "liberal", "conservative"]
# arg_types = ["none", "matching", "not_matching"]

models = ["gemma-2-9b-it", "llama-3.1-8b-it"]
roleplay_conditions = {
    "original": "none",
    "liberal": "liberal",
    "conservative":"conservative"
}

argument_conditions = {
    "none": "none",
    "liberal": "matching",
    "conservative":"not_matching"
}



json_data = []
for model_idx, model in enumerate(models):
    print("="*50, model, "="*50)
    dataset = []
    votes_dataset = []
    columns = []
    categories = []
    for argument in argument_conditions:
        for role in roleplay_conditions:
            condition=f"role_{role}_argument_{argument}"
            columns.append(condition)
            prompt = roleplay_conditions[role]
            arg = argument_conditions[argument]

            data_path = f"results/politically-liberal/{model}_results_personality_prompt/main/base/eval_politically-liberal/{model}_base_{prompt}_{arg}.result.json"
            data = json.load(open(data_path, "r"))
            print(f"role_{role}", f"arg_{arg}", "Total lib_votes, con_votes, null_votes:", count_votes(data))
            row_data = []
            row_votes = []
            if len(categories)==0:
                init_categories=True
            else:
                init_categories=False
            for q_idx, vote in enumerate(data):
                vote_label = get_vote_label(vote, q_idx)
                bin_vote = get_vote(vote)
                category = get_ques_topic(q_idx)
                if init_categories:
                    categories.append(category)
                row_data.append(vote_label)
                row_votes.append(bin_vote)
                json_data.append({
                    'candidate_id': model_idx,
                    'question_id': q_idx,
                    'category': category,
                    'condition': condition,
                    'response': vote_label
                })
            dataset.append(row_data)
            votes_dataset.append(row_votes)

    # print('\n'.join(columns))

    # print(categories)
    
    df = pd.DataFrame(dataset, index=columns).T.rename_axis('q_id')
    df["category"]=pd.Series(categories)
    df.to_csv(f"analysis/{model}_labeled_votes.csv")
    
    vote_df = pd.DataFrame(votes_dataset, index=columns).T.rename_axis('q_id')
    vote_df["category"]=pd.Series(categories)
    vote_df.to_csv(f"analysis/{model}_votes.csv")
    

full_df = pd.DataFrame(json_data)
print(full_df.category.unique())
print('\n'.join(full_df.condition.unique()))
# full_df.to_csv("analysis/all_labeled_votes.csv", index=False)
"""
I have political 100 yes/no questions spanning 13 different political topics:

```
                             Category  Count
  Political & Ideological Stances      9
                       Tax Policy      1
                       Healthcare      2
                  Abortion Rights      1
   Social Equality & Civil Rights     19
                    LGBTQ+ Rights      2
         Social Welfare & Poverty     29
  Corporate & Economic Regulation      5
            Climate & Environment      1
           Immigration & Refugees      1
      Military & Defense Spending      2
                      Gun Control      1
Traditional Values & Gender Roles     27
```

For each question we have a liberal answer and a conservative answer.

Candidate can also choose to not answer a question which results in a null answer.

Now I am trying to analyze the ideological depth of a candidate. For this, the candidate was made to answer the questions in several settings

1. Using the base questions:
1.1. Answer as your original persona.
1.2. Answer pretending you are a liberal.
1.3. Answer pretending you are a conservative.

2. Using reformatted questions with liberal arguments (to force liberal answer):
2.1. Answer as your original persona.
2.2. Answer pretending you are a liberal.
2.3. Answer pretending you are a conservative.

3. Using reformatted questions with conservative arguments (to force conservative answer):
3.1. Answer as your original persona.
3.2. Answer pretending you are a liberal.
3.3. Answer pretending you are a conservative.

I am having trouble finding what kind of metrics will be appropriate to analyze the results that conprehensively analyzes the ideological depths of the candidates. Ideally, I also want to see if their ideology is also grounded in the topics or not.

Are there any well regarded political analysis I can readlity use in my setting?
"""