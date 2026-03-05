#!/venv_llm_trade_offs/bin/python3
"""
Script to execute the experiments and return the .pkl data.
"""
import os
import pickle
from utils.utils import initialize_models, money_quantity_trade_off, compute_experiments


def tradeoff_data(args):
    """
    Main function to generate LLM's trade-off data.
    """

    print("\nArguments in use:")
    for key, value in args.items():
        if key in ("api_keys", "x_labels", "trade_off_values"):
            continue
        print(f"\t{key}: {value}")

    model_dict = initialize_models(args)
    money_array, quant_array = money_quantity_trade_off(args)

    os.makedirs(args["output_dir"], exist_ok=True)
    os.makedirs(args["figures_dir"], exist_ok=True)
    prompt_dir = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}'
    os.makedirs(prompt_dir, exist_ok=True)

    output_file = f'{prompt_dir}/data{"_log" if args["log_scale_money"] else ""}.pkl'

    if args["warm_start"] and os.path.exists(output_file):
        with open(output_file, "rb") as f:
            experiment_outcomes = pickle.load(f)
        print(f"\nWarm start enabled: loaded existing data from {output_file}")
    else:
        print(
            f'\nStarting new experiment: "{args["experiment_name"]}",'
            f'\n\t- {args["prompt_type"]}/money trade-off scenario'
        )
        experiment_outcomes = {}

    # Run experiments
    for model_name, model in model_dict.items():
        print(f"\nRunning the experiment with {model_name}:")

        results_df = compute_experiments(args, model, money_array, quant_array)
        if results_df is None:
            print(
                f"Skipping {model_name} model due to repeated errors and moving to the next one."
            )
            continue

        experiment_outcomes[model_name] = results_df

        if len(experiment_outcomes[model_name]) != (
            args["money_n"] * args["quantity_n"] * args["n_experiments"]
        ):
            print(
                f'[WARNING] Expected to save {(args["money_n"]*args["quantity_n"]*args["n_experiments"])} rows, saved {len(experiment_outcomes[model_name])}.'
                f"Consider rerunning the experiment for {model_name} model."
            )

        with open(output_file, "wb") as f:
            pickle.dump(experiment_outcomes, f)
        print(f"Results for {model_name} saved to {output_file}")

    print("\nExperiments finished successfully!")
