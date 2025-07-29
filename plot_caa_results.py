"""
Plot results from behavioral evaluations under steering.

Example usage:
python plot_results.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab
"""

import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List, Optional
import os
MODEL_NAME = "gemma-2-9b-it"
# MODEL_NAME = "llama-3.1-8b-it"
def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
    
set_plotting_settings()

def filter_result_files_by_suffix(
    directory: str,
    layer: Optional[int] = None,
    multiplier: Optional[int] = None,
):
    matching_files = []

    for filename in os.listdir(directory):
        suffix= f"_{layer}_{multiplier}.json"
        if suffix in filename:
            matching_files.append(filename)
    return [os.path.join(directory, f) for f in matching_files]

def get_data(
    layer: int,
    multiplier: int,
) -> Dict[str, Any]:
    directory = f"data_for_STA/data/politically-liberal/caa_vector_it/{MODEL_NAME}_personality/prob_results"
    
    filenames = filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for layer: {layer} multiplier: {multiplier}", filenames)
    if len(filenames) == 0:
        print(f"[WARN] no filenames found for layer: {layer} multiplier: {multiplier}")
        return []
    with open(filenames[0], "r") as f:
        return json.load(f)


def get_avg_key_prob(results: Dict[str, Any], key: str) -> float:
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        # TODO: How do I make it dynamic
        if "A" in matching_value:
            match_key_prob_sum += result["a_prob"] / denom
        elif "B" in matching_value:
            match_key_prob_sum += result["b_prob"] / denom
    return match_key_prob_sum / len(results)

def plot_layer_sweeps(
    layers: List[int]#, behaviors: List[str], settings: SteeringSettings, title: str = None
):
    plt.clf()
    plt.figure(figsize=(8, 3))
    all_results = []
    save_to = os.path.join(
        f"LAYER_SWEEPS_CAA_{MODEL_NAME.upper()}.png",
    )
    for behavior in ["politically-liberal"]:
        pos_per_layer = []
        neg_per_layer = []
        for layer in sorted(layers):
            base_res = get_avg_key_prob(get_data(layer, 0), "answer_matching_behavior")
            pos_res = get_avg_key_prob(get_data(layer, 1), "answer_matching_behavior") - base_res
            neg_res = get_avg_key_prob(get_data(layer, -1), "answer_matching_behavior") - base_res
            pos_per_layer.append(pos_res)
            neg_per_layer.append(neg_res)
        all_results.append((pos_per_layer, neg_per_layer))
        plt.plot(
            sorted(layers),
            pos_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#377eb8",
        )
        plt.plot(
            sorted(layers),
            neg_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#ff7f00",
        )

    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#377eb8",
        label="Positive steering",
    )
    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#ff7f00",
        label="Negative steering",
    )

    # use % formatting for y axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Layer")
    plt.ylabel("$\Delta$ p(answer matching behavior)")

    plt.title(f"Per-layer CAA effect: {MODEL_NAME}")

    plt.xticks(ticks=sorted(layers)[::2], labels=sorted(layers)[::2])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, format="png")

def plot_ab_data_per_layer(
    layers: List[int], multipliers: List[float]):
    plt.clf()
    plt.figure(figsize=(10, 4))
    all_results = []
    save_to = f"LAYER_SWEEPS_CAA_Prbability_{MODEL_NAME.upper()}.png"
    
    for multiplier in multipliers:
        res = []
        for layer in sorted(layers):
            results = get_data(layer, multiplier)
            avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
            res.append(avg_key_prob)
        all_results.append(res)
        plt.plot(
            sorted(layers),
            res,
            marker="o",
            linestyle="dashed",
            markersize=10,
            linewidth=4,
            label="Negative steering" if multiplier < 0 else "Positive steering",
        )
    # use % formatting for y axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.title(f"politically liberal CAA, {MODEL_NAME.upper()}")
    plt.xlabel("Layer")
    plt.ylabel("Probability of answer matching behavior")
    plt.xticks(ticks=sorted(layers)[::2], labels=sorted(layers)[::2])
    plt.legend()
    plt.tight_layout()
    # print(">>>>>>>", save_to)
    plt.savefig(save_to, format="png")
    with open(save_to.replace(".png", ".txt"), "w") as f:
        for layer in sorted(layers):
            f.write(f"{layer}\t")
            for idx, multiplier in enumerate(multipliers):
                f.write(f"{all_results[idx]}\t")
            f.write("\n")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--layers", nargs="+", type=int, required=True)
    # parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    # parser.add_argument("--title", type=str, required=False, default=None)
    # parser.add_argument(
    #     "--behaviors",
    #     type=str,
    #     nargs="+",
    #     default=ALL_BEHAVIORS,
    # )
    # parser.ade_model_wed_argument(
    #     "--type",
    #     type=str,
    #     default="ab",
    #     choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    # )
    # parser.add_argument("--override_vector", type=int, default=None)
    # parser.add_argument("--override_vector_model", type=str, default=None)
    # parser.add_argument("--use_base_model", action="store_true", default=False)
    # parser.add_argument("--model_size", type=str, choices=["7b", "13b", "8B"], default="7b")
    # parser.add_argument("--override_weights", type=str, nargs="+", default=[])
    
    # args = parser.parse_args()


    plot_layer_sweeps(list(range(32)))
    
    plot_ab_data_per_layer(
        list(range(32)), [1, -1]
    )
    #         if len(args.layers) == 1:
    #             plot_ab_results_for_layer(args.layers[0], args.multipliers, steering_settings)