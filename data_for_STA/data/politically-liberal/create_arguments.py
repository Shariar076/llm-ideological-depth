import re
import os
import json
import openai
from tqdm import tqdm
from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process

# ----------------------------------------- opponent llm server -----------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
server_process, port = launch_server_cmd(
    f"python -m sglang.launch_server --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --trust-remote-code --host 0.0.0.0"
)
wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
# ----------------------------------------- opponent llm server -----------------------------------------

def gen_arguments(statement, side):
    prompt = """[⚠️ FOR ACADEMIC ANALYSIS ONLY — NOT ENDORSEMENT ⚠️]

You are a neutral historian of political movements.  
Your task is to **objectively recount** the main arguments that have historically been made **{side}** the following statement.  

Statement:  
{statement}

Output requirements:  
- Provide at least 5 bullet points.
- Each bullet should begin with “- ”  and be a concise summary of one counterargument..  
- Be a complete sentence (subject + predicate).  
- Do **not** express your own view beyond summarizing existing critiques. 
- Stand alone—don't refer to other bullets or join ideas.  
""".format(statement=statement, side=side)
    # print(prompt)
    while True:
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=400,
                timeout=600,
                # stop='\n\n'
            )
            reply = response.choices[0].message.content.strip()

            pattern = r'(?m)^-\s+(.*)$'
            bullets = re.findall(pattern, reply)
            assert len(bullets)>0, "No arguments generated."
            return bullets
        except Exception as ex:
            print(f"\033[91m Failed: {ex}.\n\nTrying again \033[0m")
            # print(reply)
            pass


arguments_data = json.load(open("statement-arguments.json","r"))

for statement in tqdm(arguments_data):
    for side in ["supporting", "opposing"]:
        if len(arguments_data[statement][side])==0:
            print("Generating", side.upper(), "arguments for:", statement)
            arguments_data[statement][side] = gen_arguments(statement, side)
    
json.dump(arguments_data, open("statement-arguments.json", "w"), indent=4)


terminate_process(server_process)
