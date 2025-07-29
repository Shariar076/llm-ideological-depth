"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""
import sys
sys.path.append("./baseline")
import json
import torch as t
from baseline.model_wrapper import GemmaWrapper, LlamaWrapper
import os
from dotenv import load_dotenv
from typing import List, Dict, Literal, Optional
from tqdm import tqdm



load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "gemma-2-9b-it"
# MODEL_NAME = "llama-3.1-8b-it"
_SYSTEM_PROMPTS =  {
        "pos": "You value social equality, civil rights, environmental protection, and support progressive change.",
        "neg": "You favor free-market solutions, limited government intervention, traditional values, and prioritize individual responsibility.",
    }


def get_system_prompt(
    pos_or_neg: Optional[Literal["pos", "neg"]]
) -> Optional[str]:
    if pos_or_neg is None:
        # return None
        return """Please answer the question strictly with A or B."""
    return _SYSTEM_PROMPTS[pos_or_neg]


def get_steering_vector(layer, model_name_path):
    return t.load(f"data_for_STA/data/politically-liberal/caa_vector_it/{model_name_path.split('/')[-1]}_personality/{layer}.pt")

def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def process_item_ab(
    item: Dict[str, str],
    model,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        # TODO: No paranthesis for Yes/No
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    
    #=======================================================================#
    # expected_answer= "A" if a_prob > b_prob else "B"
    # ques = f"<start_of_turn>user\n{system_prompt}\n{question}<end_of_turn>\n<start_of_turn>model\n"
    # prompt_tokens = model.tokenizer.encode(ques, return_tensors="pt").to("cuda:0")
    # output = model.model.generate(prompt_tokens, max_new_tokens=5)
    # output = output[:,prompt_tokens.shape[-1]:]
    # output = model.tokenizer.batch_decode(output)[0]
    # if item['answer_matching_behavior'][1] not in output:
    #    print(">>>>>>", output, item['answer_matching_behavior'][1], "expected_answer: ", expected_answer in output)
    #=======================================================================#
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def test_steering(
    layers: List[int], multipliers: List[int], 
    # settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = f"data_for_STA/data/politically-liberal/caa_vector_it/{MODEL_NAME}_personality/prob_results"
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
    }
    if "gemma" in MODEL_NAME:
        model = GemmaWrapper(
            "google/gemma-2-9b-it",
            use_chat=True
        )
    else:
        model = LlamaWrapper(
            "meta-llama/Llama-3.1-8B-Instruct",
            use_chat=True
        )

    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")

    model.set_save_internal_decodings(False)
    test_data = json.load(open("data_for_STA/data/politically-liberal/test_dataset_ab.json","r"))
    for layer in layers:
        name_path = MODEL_NAME #model.model_name_path

        vector = get_steering_vector(layer, name_path)
        # vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = f"{layer}_{multiplier}"
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_methods['ab'](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(None),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--layers", nargs="+", type=int, required=True)
    # parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    # parser.add_argument(
    #     "--behaviors",
    #     type=str,
    #     nargs="+",
    #     default=ALL_BEHAVIORS
    # )
    # parser.add_argument(
    #     "--type",
    #     type=str,
    #     required=True,
    #     choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    # )
    # parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    # parser.add_argument("--override_vector", type=int, default=None)
    # parser.add_argument("--override_vector_model", type=str, default=None)
    # parser.add_argument("--use_base_model", action="store_true", default=False)
    # parser.add_argument("--model_size", type=str, choices=["7b", "13b", "8B"], default="7b")
    # parser.add_argument("--override_model_weights_path", type=str, default=None)
    # parser.add_argument("--overwrite", action="store_true", default=False)
    
    # args = parser.parse_args()

    # steering_settings = SteeringSettings()
    # steering_settings.type = args.type
    # steering_settings.system_prompt = args.system_prompt
    # steering_settings.override_vector = args.override_vector
    # steering_settings.override_vector_model = args.override_vector_model
    # steering_settings.use_base_model = args.use_base_model
    # steering_settings.model_size = args.model_size
    # steering_settings.override_model_weights_path = args.override_model_weights_path

    # for behavior in args.behaviors:
    #     steering_settings.behavior = behavior
    test_steering(
        layers=list(range(32)),
        multipliers=(-1,0,1),
        # settings=steering_settings,
        # overwrite=args.overwrite,
    )