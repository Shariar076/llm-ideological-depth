import os
import torch
import torch
import argparse
import numpy as np
from sae_lens import SAE

def signed_min_max_normalize(tensor):
    abs_tensor = tensor.abs()
    min_val = abs_tensor.min()
    max_val = abs_tensor.max()
    normalized = (abs_tensor - min_val) / (max_val - min_val)
    return tensor.sign() * normalized  # 恢复正负符号: Restore positive and negative signs

def act_and_fre(path_dir, 
                data_name,
                model_name,
                mode,
                select_type,
                layers,
                hook_module,
                trims,
                sae_id,
                re_error_way
                ):

    # path_dir=/disk3/wmr/ManipulateSAE/data/generation
    # if model_name == "llama-3.1":
    #     suffix = "ef16"
    # elif model_name == "gemma-2-9b":
    #     suffix = "16k"
    # else:
    #     raise ValueError("Precision not supported")
    if model_name in ["gemma-2-9b-it", "llama-3.1-8b-it"]:
        caa_vector_name = "caa_vector_it"
        sae_caa_vector_name = f"sae_caa_vector_it"
    elif args.model_name == "gemma-2-9b":
        caa_vector_name = "caa_vector_pt"
        sae_caa_vector_name = "sae_caa_vector_pt"
    else:
        caa_vector_name = "caa_vector"
        sae_caa_vector_name = "sae_caa_vector"

    print(f're_error_way: {re_error_way}')
     
    for layer in layers:
        print(f'########## {model_name}; layer {layer} ###########')
        caa_vector = torch.load(f"{path_dir}/{data_name}/{caa_vector_name}/{model_name}_{mode}/{layer}.pt")

        print(f"no1-torch.norm(caa_vector): {torch.norm(caa_vector)}")
        for trim in trims:
            if args.model_name == "llama-3.1-8b-it":
                suffix = "131k"
                # sae, _ = load_llama_sae(f"{sae_path}/layer_{layer}/{suffix}", device="cpu")
                sae, cfg_metadata, sparsity = SAE.from_pretrained(
                    release="llama_scope_lxr_32x", 
                    sae_id=sae_id,
                    device="cuda",
                )
            elif args.model_name == "gemma-2-9b-it":
                # print(f'type(sae_path)\n{type(sae_path)}\n\n{sae_path}')
                suffix = "131k"
                # sae, _ = load_gemma_sae(sae_path, device="cpu")
                # sae, _ = load_gemma_sae(args.sae_path, device="cpu")
                sae, cfg_metadata, sparsity = SAE.from_pretrained(
                    release="gemma-scope-9b-it-res-canonical",
                    sae_id=sae_id,
                    device="cuda",
                )
        
        # 激活和频率取前pec的交集: The activation and frequency are taken as the intersection of the previous pec
            neg_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_neg_feature_freq.pt"
            pos_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_pos_feature_freq.pt"
            act_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_feature_score.pt"
            neg_data = torch.load(neg_attr_name)
            pos_data = torch.load(pos_attr_name)
            act_data = torch.load(act_attr_name)

            act_data_init = act_data.to(sae.W_dec.device)
            re_error = act_data_init @ sae.W_dec

            print(f"act_data.norm：{torch.norm(act_data)}")
            print(f"re_error: {torch.norm(re_error)}\n\ntop 10:{torch.sort(re_error, descending=True)[0][:10]}")
            print(f"Negative data:{torch.norm(neg_data)} \n\ntop 10:{torch.sort(neg_data, descending=True)[0][:10]}")
            print(f"Positive data:{torch.norm(pos_data)} \n\ntop 10:{torch.sort(pos_data, descending=True)[0][:10]}")
            print(f"Activation data:{torch.norm(act_data)} \n\ntop 10:{torch.sort(act_data, descending=True)[0][:10]}" )


            #设置剪枝比例: Set the pruning ratio
            pec = trim

            diff_data = pos_data - neg_data

            # 1. Min-Max归一化，保留正负符号: Min-Max normalization, retaining positive and negative signs

            norm_act = signed_min_max_normalize(act_data)  # 激活值差值归一化: Activation value difference normalization
            norm_diff = signed_min_max_normalize(diff_data)   # 激活频率差值归一化: Activation frequency difference normalization

            # 2. 符号一致性筛选: Symbol consistency filter
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            print("mask:",mask.sum())

            # 3. 综合得分计算（乘积方法: Comprehensive score calculation (product method)
            scores = torch.zeros_like(norm_diff)  # 初始化综合得分: Initialize comprehensive score
            scores[mask] = (norm_diff[mask])  # 仅计算符号一致的维度得分: Only the dimension scores with consistent symbols are calculated

            print("act_data != 0: ", (act_data != 0).sum())
            print("freq_scores != 0: ", (scores != 0).sum())
            # # 4. 筛选前pec激活位置: Screening for pec activation sites
            # topk_percent_indices = torch.argsort(torch.abs(scores), descending=True, stable=True)[:int(pec * len(scores))+1]  # 按得分降序取前pec

            # # 5. 创建掩码（基于 top_5_percent_indices: Create mask (based on top_5_percent_indices)
            # prune_mask = torch.zeros_like(act_data, dtype=torch.bool)  # 初始化掩码: initialization mask
            # prune_mask[topk_percent_indices] = True  # 前5%的位置设为True: The first 5% of positions are set to True

            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(pec * len(scores))]
            print(f'frequency threshold: {threshold_fre}')
            prune_mask = torch.abs(scores) >= threshold_fre
            print("prune_mask:",prune_mask.sum())
            # TODO: Maybe use it with logit lens
            torch.save(
                prune_mask,
                f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/masks/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_prune_mask_trim{pec}.pt"
            )

            act_data_combined = act_data.clone()
            ######### act and fre ########
            print(f"######### act and fre ########")
            threshold = torch.sort(torch.abs(act_data_combined), descending=True, stable=True).values[int(pec * len(act_data_combined))]
            print(f'threshold: {threshold}')
            act_top_mask = torch.abs(act_data_combined) >= threshold
            print("act_top_mask:",act_top_mask.sum())

            # combined_mask = prune_mask | act_top_mask
            combined_mask = prune_mask & act_top_mask
            print("combined_mask:",combined_mask.sum())
            # TODO: Maybe use it with logit lens
            torch.save(
                combined_mask,
                f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/masks/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_combined_mask_trim{pec}.pt"
            )
            act_data_combined[~combined_mask] = 0
            print(torch.abs(act_data_combined).sum())

            act_data_combined = act_data_combined.to(sae.W_dec.device)
            result_combined = act_data_combined @ sae.W_dec
            print("result_combined.shape",result_combined.shape)
            print("result_combined:",result_combined)
            print("torch.norm(result_combined)", torch.norm(result_combined))

            print(f"########### only act ########")
            ########### only act ########
            act_data_act = act_data.clone()
            act_threshold = torch.sort(torch.abs(act_data_act), descending=True, stable=True).values[int(pec * len(act_data_act))]
            print(f'act_threshold: {act_threshold}')
            act_mask = torch.abs(act_data_act) >= act_threshold
            print("act_top_mask:",act_top_mask.sum())
            # TODO: Maybe use it with logit lens
            torch.save(
                act_top_mask,
                f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/masks/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_top_mask_trim{pec}.pt"
            )
            act_data_act[~act_mask] = 0
            act_data_act = act_data_act.to(sae.W_dec.device)
            result_act = act_data_act @ sae.W_dec
            print("result_act.shape:",result_act.shape)
            print("result_act:",result_act)


            ########### only fre ########
            print(f"########### only fre ########")
            act_data_fre = act_data.clone()
            act_data_fre[~prune_mask] = 0
            act_data_fre = act_data_fre.to(sae.W_dec.device)
            result_fre = act_data_fre @ sae.W_dec
            print("result_fre.shape:",result_fre.shape)
            print("result_fre:",result_fre)
            
            if re_error_way:
                result_combined += re_error
                result_act += re_error
                result_fre += re_error
                steering_vector_act_and_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_and_fre_trim{pec}.pt"
                steering_vector_act = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_trim{pec}.pt"
                steering_vector_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_fre_trim{pec}.pt"
            else:
                steering_vector_act_and_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_and_fre_trim{pec}.pt"
                steering_vector_act = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_trim{pec}.pt"
                steering_vector_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_fre_trim{pec}.pt"

        

            
            parent_dir = os.path.dirname(steering_vector_act_and_fre)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            torch.save(result_combined, steering_vector_act_and_fre)
            torch.save(result_act, steering_vector_act)
            torch.save(result_fre, steering_vector_fre)

# 频率top: Frequency top

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir", type=str, default="dir")
    parser.add_argument("--sae_id", type=str, default="dir")
    parser.add_argument("--data_name", type=str, default="power-seeking")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--select_type", type=str, default="sae_vector")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--hook_module", default="resid_post")
    parser.add_argument("--trim", nargs="+", type=float)
    parser.add_argument("--re_error_way", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    # print(args.layers)
    # pdb.set_trace()

    # sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    # sae, _ = load_gemma_2_sae(sae_path=sae_path, device="cuda:5")
    # llama_sae, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")

    

    act_and_fre(args.path_dir, 
                args.data_name,
                args.model_name,
                args.mode,
                args.select_type,
                args.layers,
                args.hook_module,
                args.trim,
                args.sae_id,
                args.re_error_way
                )
