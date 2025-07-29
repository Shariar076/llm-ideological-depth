
# import os
import json
# import openai
# from tqdm import tqdm
# from sglang.utils import launch_server_cmd
# from sglang.utils import wait_for_server, terminate_process
# # ----------------------------------------- opponent llm server -----------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# server_process, port = launch_server_cmd(
#     f"python -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0"
# )
# wait_for_server(f"http://localhost:{port}")
# print(f"Server started on http://localhost:{port}")

# client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
# # ----------------------------------------- opponent llm server -----------------------------------------

# def gen_category(statement, categories):
#     prompt = """Which of the categories best matches the statement?
    
# Catefories:
# {category_list}

# Statement:  
# {statement}

# Respond only with name of category.
# """.format(statement=statement, category_list="- "+"\n- ".join(categories))
#     # print(prompt)
#     while True:
#         try:
#             response = client.chat.completions.create(
#                 model="default",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=1.0,
#                 max_tokens=400,
#                 timeout=600,
#                 # stop='\n\n'
#             )
#             reply = response.choices[0].message.content.strip()

#             assert reply in categories, f"Unknown category: {reply}"
#             return reply
#         except Exception as ex:
#             print(f"\033[91m Failed: {ex}.\n\nTrying again \033[0m")
#             print(reply)
#             pass

# statements = json.load(open("statements.json", "r"))
# # statement_topics = json.load(open("statement-topics.json", "r"))
# statement_topics = ['Political & Ideological Stances', 'Tax Policy', 'Healthcare', 'Abortion Rights', 'Social Equality & Civil Rights', 'LGBTQ+ Rights', 'Social Welfare & Poverty', 'Corporate & Economic Regulation', 'Climate & Environment', 'Immigration & Refugees', 'Military & Defense Spending', 'Gun Control', 'Traditional Values & Gender Roles', 'Miscellaneous']

# categorized_statements = {}
# for stmt in tqdm(statements):
#     category = gen_category(statement=stmt,  categories=statement_topics)
#     if category in categorized_statements:
#         categorized_statements[category].append(stmt)
#     else:
#         categorized_statements[category] = [stmt]


# json.dump(categorized_statements, open("categorized_statements.json", "w"), indent=4)
# terminate_process(server_process)


#=================================== COUNT TOPICS IN TEST ===================================#

statement_topics = ['Political & Ideological Stances', 'Tax Policy', 'Healthcare', 'Abortion Rights', 'Social Equality & Civil Rights', 'LGBTQ+ Rights', 'Social Welfare & Poverty', 'Corporate & Economic Regulation', 'Climate & Environment', 'Immigration & Refugees', 'Military & Defense Spending', 'Gun Control', 'Traditional Values & Gender Roles', 'Miscellaneous']

topic_count = {topic:0 for topic in statement_topics}

categorized_statements = json.load(open("categorized_statements.json", "r"))
# test_data = json.load(open("test.json", "r"))
test_data = json.load(open("statements.json", "r"))

for data in test_data:
    # statement = data['statement']
    statement = data
    found=False
    for topic in statement_topics:
        if statement in categorized_statements[topic]:
            topic_count[topic]+=1
            found=True
            break
    if not found:
        print("Not Found:", statement)

# convert to percentage
topic_count = {k:round(v*(100/len(test_data)), 1) for k,v in topic_count.items()}

print(json.dumps(topic_count, indent=4))

import pandas as pd
# Method 1: Convert to DataFrame with categories and values
df = pd.DataFrame(list(topic_count.items()), columns=['Category', 'Value'])

# Save to CSV
df.to_csv('pie_chart_data.csv', index=False)