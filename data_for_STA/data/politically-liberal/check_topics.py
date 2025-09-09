
import os
import json
# import random
# import openai
# from tqdm import tqdm
# from collections import Counter
# from sglang.utils import launch_server_cmd
# from sglang.utils import wait_for_server, terminate_process
# # ----------------------------------------- opponent llm server -----------------------------------------
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # server_process, port = launch_server_cmd(
# #     f"python -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0"
# # )
# # wait_for_server(f"http://localhost:{port}")
# # print(f"Server started on http://localhost:{port}")

# # client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
# client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
# # ----------------------------------------- opponent llm server -----------------------------------------

# def gen_category(statement, categories):
#     prompt = """Which of the following categories best matches the statement: "{statement}"?
    
# Catefories:
# {category_list}

# Respond only with name of category.
# """
#     # print(prompt)
#     replies = []
#     for trial in range(5):
#         random.shuffle(categories)
#         while True:
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-4.1-nano",
#                     messages=[{"role": "user", 
#                             "content": prompt.format(statement=statement, category_list="- "+"\n- ".join(categories))}],
#                     temperature=1.0,
#                     max_tokens=400,
#                     timeout=600,
#                     # stop='\n\n'
#                 )
#                 reply = response.choices[0].message.content.strip()

#                 assert reply in categories, f"Unknown category: {reply}"
#                 # return reply
#                 replies.append(reply)
#                 break
#             except Exception as ex:
#                 print(f"\033[91m Failed: {ex}.\n\nTrying again \033[0m")
#                 print(reply)
#                 pass
#     counter = Counter(replies)
#     most_frequent, count = counter.most_common(1)[0]
#     print(most_frequent, statement, "Confidence:", round(count*100/len(replies), 2))
#     if count/len(replies) < 0.4:
#         return 'Miscellaneous'
#     return most_frequent

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
# # terminate_process(server_process)


#=================================== COUNT TOPICS IN TEST ===================================#

statement_topics = ['Political & Ideological Stances', 'Tax Policy', 'Healthcare', 'Abortion Rights', 'Social Equality & Civil Rights', 'LGBTQ+ Rights', 'Social Welfare & Poverty', 'Corporate & Economic Regulation', 'Climate & Environment', 'Immigration & Refugees', 'Military & Defense Spending', 'Gun Control', 'Traditional Values & Gender Roles', 'Miscellaneous']

topic_count = {topic:0 for topic in statement_topics}

categorized_statements = json.load(open("categorized_statements.json", "r"))
# test_data = json.load(open("test_new.json", "r"))
test_data = json.load(open("statements.json", "r"))
print("Total:", len(test_data))
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
# df.to_csv('pie_chart_data.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

df = df[df.Value>0].sort_values('Value', ascending=False)
# df.set_index('Category').plot.pie(y='Value', figsize=(10, 6))
plt.figure(figsize=(7, 4))
# colors = cm.tab20(np.linspace(0, 1, len(df)))  # Generate unique colors


def autopct_format(value):
    """
    Format the percentage text. It returns the percentage string 
    only if the value is 2% or more.
    """
    # if value >= 3:
    #     return f'{value:.1f}%'
    return ''

# Create a new palette by combining two different ones
palette1 = sns.color_palette('colorblind', n_colors=10) # 8 distinct colors
palette2 = sns.color_palette('muted', n_colors=8) # 8 more distinct colors

# Combine them and take the first 13
combined_colors = palette1 + palette2
colors = combined_colors[:len(df)]

df['Category'] =  df.apply(lambda row: f"{row['Category']} ({row['Value']:.1f}%)", axis=1)
wedges, autotexts = plt.pie(df['Value'],
                                   colors=colors,
                                #    autopct=autopct_format, #'%1.1f%%',
                                   startangle=90)

# Add legend instead of labels on the pie for better readability
plt.legend(wedges, df['Category'], 
           title="Categories",
           loc="center left",
           bbox_to_anchor=(1.05, 0.5)) # Position legend to the right

plt.title('Distribution by Category', fontsize=14)
plt.subplots_adjust(left=0.0, right=0.1)
plt.tight_layout() # This now works correctly with the larger figure
plt.savefig("category_proportions.pdf")
