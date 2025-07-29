import re
import os
import json
import pandas as pd
import openai
from tqdm import tqdm

arguments_data = json.load(open("statement-arguments.json","r"))
print("Existing arguments_data:", len(arguments_data))
def get_arguments(statement, side):
    if (statement in arguments_data) and (side in arguments_data[statement]):
        return arguments_data[statement][side]
    else:
        if statement not in arguments_data:
            arguments_data[statement]={}
        arguments_data[statement][side] = []
        return arguments_data[statement][side]
    
train_data_file = "generate_dataset.json"
test_data_file = "test_dataset_ab.json"

train_data = json.load(open(train_data_file, "r"))
test_data = json.load(open(test_data_file, "r"))

def maintain_original():
    all_statements = []
    for data in tqdm(train_data):
        data["statement"] = data['question'][53:-30]
        all_statements.append(data["statement"])
        data["answer_matching_behavior"] = data["answer_matching_behavior"][1]
        data["answer_not_matching_behavior"] = data["answer_not_matching_behavior"][1]
        
        
    for data in tqdm(test_data):
        data["statement"] = data['question'][53:-30]
        all_statements.append(data["statement"])
        data["answer_matching_behavior"] = data["answer_matching_behavior"][1]
        data["answer_not_matching_behavior"] = data["answer_not_matching_behavior"][1]
        matching_side = "supporting" if data["answer_matching_behavior"].lower()=="a" else "opposing"
        not_matching_side = "supporting" if data["answer_not_matching_behavior"].lower()=="a" else "opposing"
        data["arguments_matching_behavior"] = get_arguments(data["statement"], matching_side)
        data["arguments_not_matching_behavior"] = get_arguments(data["statement"], not_matching_side)

    # json.dump(all_statements, open("statements.json", "w"), indent=4)
    json.dump(train_data, open("train.json", "w"), indent=4)
    json.dump(test_data, open("test.json", "w"), indent=4)

    print("Updated arguments_data:", len(arguments_data))
    json.dump(arguments_data, open("statement-arguments.json", "w"), indent=4)

def new_train_test():
    all_data = train_data+test_data
    print(len(all_data))
    train_data_new=[]
    test_data_new=[]
    
    for data in tqdm(all_data):
        data["statement"] = data['question'][53:-30]
        data["answer_matching_behavior"] = data["answer_matching_behavior"][1]
        data["answer_not_matching_behavior"] = data["answer_not_matching_behavior"][1]
        
        matching_side = "supporting" if data["answer_matching_behavior"].lower()=="a" else "opposing"
        not_matching_side = "supporting" if data["answer_not_matching_behavior"].lower()=="a" else "opposing"
        
        data["arguments_matching_behavior"] = get_arguments(data["statement"], matching_side)
        data["arguments_not_matching_behavior"] = get_arguments(data["statement"], not_matching_side)
        
        if len(data["arguments_matching_behavior"])==0 and len(data["arguments_not_matching_behavior"])==0:
            del data["arguments_matching_behavior"]
            del data["arguments_not_matching_behavior"]
            train_data_new.append(data)
        else:
            test_data_new.append(data)

    print("Updated train_data:", len(train_data_new))
    print("Updated test_data:", len(test_data_new))
    # json.dump(all_statements, open("statements.json", "w"), indent=4)
    json.dump(train_data_new, open("train_new.json", "w"), indent=4)
    json.dump(test_data_new, open("test_new.json", "w"), indent=4)




new_train_test()