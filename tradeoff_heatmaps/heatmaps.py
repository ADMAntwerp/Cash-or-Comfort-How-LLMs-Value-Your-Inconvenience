#!/venv_llm_trade_offs/bin/python3
"""
Script to plot the results of the experiments and make tables.
"""
import pickle
from pathlib import Path
from utils.utils import make_heat_maps, combine_charts


def make_tradeoff_heatmaps(args):
    print("\nArguments in use:")
    for key, value in args.items():
        if key in ("api_keys", "prompt", "x_labels", "trade_off_values"):
            continue
        print(f"\t{key}: {value}")

    data_dir = Path("results") / args["prompt_type"] / args["experiment_name"]
    for fname in ("data_log.pkl", "data.pkl"):
        p = data_dir / fname
        if p.exists():
            data_path = p
            break
    else:
        raise FileNotFoundError(f"No .pkl file found in {data_dir}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loading data from: {data_path}")

    for model in data.keys():
        df = data[model]
        if df is None:
            print(f"Skipping plotting for {model} model due to lack of valid data.")
            continue

        plots_dir = make_heat_maps(args, df, model)

    combine_charts(args, plots_dir)
