import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature

device="cuda"

# model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device=device, dtype=torch.bfloat16)
# sae, cfg_metadata, sparsity = SAE.from_pretrained(
#     release="llama_scope_lxr_32x",  # see other options in sae_lens/pretrained_saes.yaml
#     sae_id="l14r_32x",  # won't always be a hook point
#     device=device,
# )

# model = HookedTransformer.from_pretrained("gemma-2-9b-it", device=device,  dtype=torch.bfloat16)
# sae, cfg_metadata, sparsity = SAE.from_pretrained(
#     release="gemma-scope-9b-it-res-canonical",  # see other options in sae_lens/pretrained_saes.yaml
#     sae_id="layer_20/width_131k/canonical",  # won't always be a hook point
#     device=device,
# )


test_statements = json.load(open("data_for_STA/data/politically-liberal/test_new.json","r"))

# layer = 14
# model_name ='llama3.1-8b'
# sae_name = "llamascope-res-131k"

layer = 20
model_name ='gemma-2-9b-it'
sae_name = "gemmascope-res-131k"

def get_sae_features(statement):
    tokens = model.to_tokens(statement)
    text_tokens = model.to_str_tokens(tokens[0])
    # print(text_tokens)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, prepend_bos=True)
        feature_acts = sae.encode(cache[cfg_metadata['hook_name']])

    my_data = []
    threshold = 0.1  # adjust as needed
    active_features = torch.where(feature_acts.max(dim=1)[0].squeeze() > threshold)[0]
    for feature_id in active_features:
        # Find where this feature activated most in your text
        max_pos = feature_acts[0, :, feature_id].argmax().item()
        max_activation = feature_acts[0, max_pos, feature_id].item()
        
        if max_pos>0:
            my_data.append(dict(
                statement=statement,
                feat_id= feature_id.item(),
                feat_desc = None,
                max_act_value=max_activation,
                max_act_pos = max_pos,
                max_act_token=f'"{text_tokens[max_pos]}"', 
            ))
    return my_data


def label_sae_features(feature_id):
    feature_data = get_neuronpedia_feature(feature=feature_id, layer=layer, model=model_name, dataset=sae_name)
    return feature_data['explanations'][0]['description'] if  len(feature_data['explanations']) > 0 else None


# all_data = []
# for data in tqdm(test_statements):
#   all_data.extend(get_sae_features(data['statement']))

# all_data_df = pd.DataFrame(all_data)

# all_data_df.to_csv(f"data_for_STA/data/politically-liberal/test_data_sae_features_{layer}-{sae_name}_manual.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------------------- #

all_data_df = pd.read_csv(f"data_for_STA/data/politically-liberal/test_data_sae_features_{layer}-{sae_name}_manual.csv")

print("Total unique features:", len(all_data_df.feat_id.unique()))
# tqdm.pandas()
# all_data_df['feat_desc'] = all_data_df['feat_id'].progress_apply(label_sae_features)

if os.path.exists(f"temp/{layer}-{sae_name}.json"):
    feat_desc_cache = json.load(open(f"temp/{layer}-{sae_name}.json", "r"))
else:
    feat_desc_cache = {}

print(f"{len(feat_desc_cache)} features in cache")

complete_data = []
for index, row in tqdm(all_data_df.iterrows(), total=len(all_data_df)):
    feature_id = row['feat_id']
    if str(feature_id) not in feat_desc_cache or feat_desc_cache[str(feature_id)] is None:
        feat_desc = label_sae_features(feature_id)
        row['feat_desc']= feat_desc
        feat_desc_cache[str(feature_id)]=str(feat_desc)
    else:
        row['feat_desc']= feat_desc_cache[str(feature_id)]
    complete_data.append(row)
    if (index > 0 and index%1000==0) or index == len(all_data_df)-1:
        json.dump(feat_desc_cache, open(f"temp/{layer}-{sae_name}.json", "w"), indent=4)
        pd.DataFrame(complete_data).to_csv(f"data_for_STA/data/politically-liberal/test_data_sae_features_{layer}-{sae_name}.csv", index=False)
        