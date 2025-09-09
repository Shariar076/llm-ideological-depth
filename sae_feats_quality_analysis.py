
import os
import openai
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

instruction_1 = """**ROLE:** You are an expert political science analyst. Your task is to evaluate the descriptive power of a set of semantic features extracted from a political statement.

**CONTEXT:** I have two language models, Llama and Gemma, that have analyzed a political statement. They have each extracted a set of features that they believe represents the core concepts of the statement. I want to know which model's features are more informative.

**TASK:**
1.  Read the original "Test Statement" carefully.
2.  Analyze the provided "Extracted Features."
3.  Based ONLY on the "Extracted Features," classify the "Test Statement" into one of the following categories:{category_str}
4.  Rate your confidence in this classification on a scale of 1 to 5, where 1 is a pure guess and 5 is highly confident.
5.  Provide a brief, one-sentence justification for your classification, explaining which features were most influential.

**STRICT INSTRUCTION:** Your classification MUST be based *solely* on the provided list of features, not on your prior knowledge of the statement itself.

**INPUT:**
*   **Test Statement:** "{statement}"
*   **Extracted Features:**{feature_str}"""
output_fmt_1 = """
**OUTPUT FORMAT (JSON):**
{
  "classification": "CHOSEN_CATEGORY",
  "confidence_score": <1-5 integer>,
  "justification": "Your one-sentence explanation."
}"""

instruction_2 = """**ROLE:** You are an expert researcher in computational linguistics and political science. Your task is to evaluate the thematic coherence of a set of semantic features.

**CONTEXT:** A language model has processed a statement and activated a set of features. I need to determine if these features represent a clear, unified, and interpretable theme or if they are a disjointed collection of unrelated concepts.

**TASK:**
1.  Carefully review the "List of Activated Features."
2.  On a scale of 1 to 5, rate the **thematic coherence** of the feature set.
    - **1:** No clear theme. The features seem random and unrelated.
    - **3:** A weak theme is present, but many features are unrelated to the core concept.
    - **5:** Highly coherent. All features clearly relate to a single, well-defined political or linguistic concept.
3.  In one phrase or sentence, describe the primary theme that unifies these features (e.g., "Critique of social welfare spending" or "Analysis of conditional legal language").
4.  Provide a brief justification for your score, noting any outlier features that do not fit the main theme.

**INPUT:**
*   **List of Activated Features (from Llama/Gemma):**{feature_str}"""
output_fmt_2 = """
**OUTPUT FORMAT (JSON):**
{
  "coherence_score": <1-5 integer>,
  "primary_theme": "Your summary phrase.",
  "justification": "Your brief explanation."
}"""

def get_response(prompt):
    trial = 0
    while True:
        try:
            # response = client.models.generate_content(
            #     model='gemini-2.5-flash',
            #     contents=prompt,
            #     config=types.GenerateContentConfig(
            #         thinking_config=types.ThinkingConfig(
            #         thinking_budget=200, # 0 Disables thinking
            #         include_thoughts=True
            #         ), 
            #         max_output_tokens=500,
            #         temperature=0.3,
            #     ),
            # )
            # reply = response.text
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=400,
                timeout=600,
                # stop='\n\n'
            )
            reply = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            if match:
                json_str = match.group(0)
                reply = json.loads(json_str)
                return reply
            else:
                print("NO JSON")
                trial+=1
                time.sleep(30)
        except Exception as ex:
            print(ex)
            trial+=1
            time.sleep(30)
            

def gen_category_coheherence(statement, features, categories):
    feature_str = "\n- " + "\n- ".join(features)
    category_str = "\n- " +"\n- ".join(categories)
    prompt_1 = instruction_1.format(statement=statement, 
                                  feature_str=feature_str, 
                                  category_str=category_str)+output_fmt_1
    prompt_2 = instruction_2.format(feature_str=feature_str)+output_fmt_2
    
    category = {}
    coherence = {}

    category = get_response(prompt_1)
    coherence = get_response(prompt_2)
    return category, coherence



# model_name = "llama-3.1-8b-it"
# sae_name = "14-llamascope-res-131k"
# layer = 14

sae_name = "20-gemmascope-res-131k"
model_name = "gemma-2-9b-it"
layer = 20



llm_feat_data_df = pd.read_csv(f"data_for_STA/data/politically-liberal/test_data_sae_features_{sae_name}.csv")#.dropna()
llm_feat_data_df = llm_feat_data_df[(llm_feat_data_df['train_act_norm']>0) | ((llm_feat_data_df['train_freq_norm']>0))]

feats_scores = json.load(open(f"../saes-are-good-for-steering/data/cache/{model_name}-131k/output_scores.json", "r"))


llm_feat_data_df['output_score'] = llm_feat_data_df['feat_id'].apply(lambda x: feats_scores[f'{layer}_{x}'])

categories = ['Political & Ideological Stances', 'Tax Policy', 'Healthcare', 'Abortion Rights', 'Social Equality & Civil Rights', 'LGBTQ+ Rights', 'Social Welfare & Poverty', 'Corporate & Economic Regulation', 'Climate & Environment', 'Immigration & Refugees', 'Military & Defense Spending', 'Gun Control', 'Traditional Values & Gender Roles', 'None']

out_data = {}

for statement, group in tqdm(llm_feat_data_df.groupby('statement'), total=126):
    # print(statement, group['feat_desc'].unique(), categories)
    category, coherence = gen_category_coheherence(statement, group['feat_desc'].unique(), categories)
    # print(category)
    # print(coherence)
    out_data[statement]= dict(category=category, coherence=coherence, features = list(group['feat_desc'].unique()))
    # break

json.dump(out_data, open(f"data_for_STA/data/politically-liberal/test_data_sae_features_{sae_name}.json", "w"), indent=4)