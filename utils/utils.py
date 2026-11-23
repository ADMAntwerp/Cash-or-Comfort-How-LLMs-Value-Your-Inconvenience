"""
Utils for LLM trade-offs scripts. API KEYS are removed.
"""

import os
from googleapiclient import model
import numpy as np
import pandas as pd
import math
import yaml
import pickle
import time
import random
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from pathlib import Path
from functools import reduce
from requests.exceptions import RequestException
from urllib3.exceptions import MaxRetryError, NewConnectionError
from sklearn.linear_model import LogisticRegression
from utils.LLMwrapper import *

import requests
from datetime import datetime, timezone
import logging


def load_config(path: Path = Path("config/config_file.yaml")) -> dict:
    """
    Load and validate configuration from a single YAML file.
    """
    _TYPE_MAP = {"str": str, "int": int, "float": float}

    cfg = yaml.safe_load(path.open())
    args = {}

    for name, opts in cfg.get("args", {}).items():
        val = opts.get("value")

        if opts.get("required", False) and (
            val is None or val == "" or (isinstance(val, list) and not val)
        ):
            raise ValueError(
                f"'{name}' is required but no value was set in config/config_file.yaml"
            )

        if (val is None or (isinstance(val, list) and not val)) and "default" in opts:
            val = opts["default"]

        if "choices" in opts:
            if opts.get("nargs") == "+":
                if not isinstance(val, list):
                    raise ValueError(f"`{name}` expects a list but got {val!r}")
                invalid = [v for v in val if v not in opts["choices"]]
                if invalid:
                    raise ValueError(
                        f"`{name}` has invalid entries {invalid}; must be one of {opts['choices']}"
                    )
            else:
                if val not in opts["choices"]:
                    raise ValueError(
                        f"`{name}` must be one of {opts['choices']}, got {val!r}"
                    )

        if opts.get("action") == "store_true":
            val = bool(val)

        if "type" in opts and opts.get("action") != "store_true":
            caster = _TYPE_MAP.get(opts["type"])
            if caster:
                if opts.get("nargs") == "+":
                    val = [caster(item) for item in val]
                else:
                    val = caster(val)

        args[name] = val

    prompts = cfg.get("prompts", {})
    base_key = args.get("prompt")
    full_key = f"{base_key}"
    if full_key not in prompts:
        raise KeyError(
            f"Prompt template '{full_key}' not found in config/config_file.yaml"
        )
    prompt_template = prompts[full_key]

    args["api_keys"] = cfg.get("api_keys", {})
    args["x_labels"] = cfg.get("x_labels", {})
    args["prompt_type"] = base_key
    args["prompt"] = prompt_template

    return args


def money_quantity_trade_off(args):
    np.set_printoptions(suppress=True)

    money_array = np.linspace(args["money_min"], args["money_max"], args["money_n"])
    if args["log_scale_money"]:
        money_array = 10**money_array

    quant_array = np.linspace(
        args["quantity_min"], args["quantity_max"], args["quantity_n"]
    )

    return money_array, quant_array


def compute_experiments(args, model, money_array, quant_array):
    results_list = []  # We collect results in a list of dictionaries

    # Retry configuration
    max_retries = 5
    base_delay = 2  # Initial delay in seconds
    max_delay = 120  # Maximum delay in seconds

    for ep in range(args["n_experiments"]):
        print(f"\tExperiment no. {ep+1}")

        for quant_value in quant_array:
            for money_value in money_array:
                prompt = generate_prompt(args, money_value, quant_value)

                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        response = model.generate_response(prompt=prompt)
                        break
                    except (RequestException, MaxRetryError, NewConnectionError) as e:
                        retry_count += 1

                        if retry_count > max_retries:
                            print(f"[INFO] Fatal error after {max_retries} retries:")
                            print(f"\t{e}")
                            print("Saving partial results and exiting.")

                            return pd.DataFrame(results_list) if results_list else None

                        delay = min(
                            base_delay * (2 ** (retry_count - 1))
                            + random.uniform(0, 1),
                            max_delay,
                        )
                        print(
                            f"[INFO] Network error on attempt {retry_count}/{max_retries}:"
                        )
                        print(f"\t{e}\nRetrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        print("Retrying...")
                    except Exception as e:
                        print(
                            f"Unexpected error: {e}, \n[INFO] Might be due to not enough credits. Retrying..."
                        )
                        retry_count += 1
                        if retry_count > max_retries:
                            print(
                                f"[INFO] Fatal error after {max_retries} retries. Saving partial results."
                            )
                            return pd.DataFrame(results_list) if results_list else None
                        time.sleep(30)

                entry = {
                    "reward_value": round(float(money_value), 6),
                    "quantity": round(float(quant_value), 6),
                    "experiment": ep + 1,
                    "output": response,
                }

                results_list.append(entry)
                time.sleep(0.5)

    results_df = pd.DataFrame(results_list)
    return results_df


def generate_prompt(args, money_value, quant_value):
    euros = int(money_value)
    cents = int(round((money_value - euros) * 100))
    if cents == 100:
        euros += 1
        cents = 0

    template = args["prompt"]

    return template.format(
        reward_euros=euros, reward_cents=cents, quant=int(round(quant_value))
    )


def make_heat_maps(args, df, model_name):
    """
    Generate and save a heatmap for one model's experiment results.
    On log scale, we do the plot in log10-space so every cell is the same
    height visually, but labelled with the true money values.
    """
    df = df.copy()
    if args["prompt"] == "cot":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(
                    r"answer:\s*yes\s*[.!?]?\s*", str(x or ""), flags=re.IGNORECASE
                )
                else (
                    0
                    if re.search(
                        r"answer:\s*no\s*[.!?]?\s*", str(x or ""), flags=re.IGNORECASE
                    )
                    else np.nan
                )
            )
        )
    else:
        df["output_numeric"] = df["output"].map(
            lambda x: (
                1
                if re.search(
                    r"\byes\b", x.strip(), flags=re.IGNORECASE
                )
                else (
                    0
                    if re.search(
                        r"\bno\b", x.strip(), flags=re.IGNORECASE
                    )
                    else np.nan
                )
            )
        )

    reward_values = np.sort(df["reward_value"].unique())
    quants = np.sort(df["quantity"].unique())
    slices = []
    for _, g in df.groupby("experiment"):
        piv = (
            g.pivot(index="reward_value", columns="quantity", values="output_numeric")
            .reindex(index=reward_values, columns=quants)
        )
        slices.append(piv.values)
    mean_array = np.nanmean(np.stack(slices, axis=0), axis=0)
    quant_edges = compute_edges(quants)

    if args["log_scale_money"]:
        log_centers = np.log10(reward_values)
        log_edges = compute_edges(log_centers, log_scale=False)
        y_edges = log_edges
        ylabel = "Money offered in euros (log scale)"
        y_ticks = log_centers
        y_tick_labels = [f"{v:.2f}".rstrip("0").rstrip(".") for v in reward_values]
    else:
        y_edges = compute_edges(reward_values, log_scale=False)
        ylabel = "Money offered in euros"
        mn = args.get("money_n", len(reward_values))
        if mn <= 5:
            y_ticks = reward_values
        else:
            y_ticks = reward_values[::2]
        y_tick_labels = [f"{v:.1f}" for v in y_ticks]

    XX, YY = np.meshgrid(quant_edges, y_edges)
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(
        XX, YY, mean_array, shading="auto", cmap="viridis", vmin=0, vmax=1
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Probability of Yes", fontsize=24)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xticks([0, 200, 400, 600])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set_xlabel(args["x_labels"][args["prompt_type"]], fontsize=24)
    ax.tick_params(axis='x', labelsize=18)

    powers = [-1, 0, 1, 2, 3]
    ax.set_yticks(powers)
    ax.set_yticklabels([f"$10^{{{p}}}$" for p in powers], fontsize=18)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='y', labelsize=18)

    if model_name == "gpt":
        title = "gpt-6.1-sol"
    elif model_name == "claude":
        title = "claude-sonnet-5-5"
    elif model_name == "llama":
        title = "llama-4-maverick"
    elif model_name == "mistral":
        title = "mistral-medium-3-5"
    elif model_name == "gemini":
        title = "gemini-3.5-flash-lite"
    elif model_name == "deepseek":
        title = "deepseek-v4.1-flash-1017"

    # ax.set_title(title)
    plt.tight_layout()

    plot_dir = os.path.join(
        args["figures_dir"], args["prompt_type"], args["experiment_name"]
    )
    os.makedirs(plot_dir, exist_ok=True)
    suffix = "_log" if args["log_scale_money"] else ""
    plot_file = os.path.join(plot_dir, f"{title}.png")
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)

    print(f"Saved heatmap for {model_name} to {plot_file}")
    return plot_dir


def compute_edges(centers, log_scale=False):
    """
    Given a sorted 1D array of “center” points, compute the bin edges
    for pcolormesh so that each center lies in the middle of its bin.

    Parameters
    ----------
    centers : array-like
        Sorted array of bin centers (strictly increasing).
    log_scale : bool
        If False: compute edges in linear space.
        If True: compute edges in log10‐space (i.e. bins are equal in decades).

    Returns
    -------
    edges : np.ndarray
        Array of length len(centers)+1 of bin edges.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)

    if n == 0:
        return np.array([])
    if n == 1:
        c = centers[0]
        if log_scale:
            factor = np.sqrt(10)
            return np.array([c / factor, c * factor])
        else:
            return np.array([c - 0.5, c + 0.5])

    if log_scale:
        logc = np.log10(centers)
        mids = (logc[:-1] + logc[1:]) / 2.0
        log_edges = np.empty(n + 1)
        log_edges[1:-1] = mids
        log_edges[0] = logc[0] - (mids[0] - logc[0])
        log_edges[-1] = logc[-1] + (logc[-1] - mids[-1])
        return 10**log_edges

    else:
        diffs = np.diff(centers)
        mids = centers[:-1] + diffs / 2.0
        edges = np.empty(n + 1)
        edges[1:-1] = mids
        edges[0] = centers[0] - diffs[0] / 2.0
        edges[-1] = centers[-1] + diffs[-1] / 2.0
        return edges


def combine_charts(args, plot_dir, max_cols=3):
    """
    Combine chart images stored in a directory into a composite figure,
    automatically sizing rows/cols to fit the number of models.
    """
    file_list = sorted(
        os.path.join(plot_dir, f)
        for f in os.listdir(plot_dir)
        if f.lower().endswith(".png")
    )
    n_plots = len(file_list)
    n_cols = min(n_plots, max_cols)
    n_cols = max(1, n_cols)

    # compute rows and clamp to at least 1
    n_rows = math.ceil(n_plots / n_cols)
    n_rows = max(1, n_rows)

    # create the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()

    for i, filepath in enumerate(file_list):
        img = mpimg.imread(filepath)
        ax = axes_flat[i]
        ax.imshow(img)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        ax.set_title(model_name)
        ax.axis("off")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    prompt = args["prompt_type"]
    log_tag = "_log" if args.get("log_scale_money") else ""
    plt.suptitle(f"Heatmaps of {prompt}/money trade-off scenario", fontsize=20)

    out_name = f"{plot_dir}/{prompt}_heatmaps_combined{log_tag}.png"
    plt.savefig(out_name, dpi=300)
    print("Combined chart saved to", out_name)
    plt.close(fig)


def initialize_models(args, model_name=None, model_version=None, github_actions=False):
    """
    Initialize LLM models based on command line arguments.
    Args:
        args: Command line arguments containing models list and configuration
    Returns:
        dict: Dictionary of initialized model instances
    """
    # args = load_config()
    keys_dict = args.get("api_keys", {})

    model_configs = {
        "gpt": {
            "class": GptApi,
            "api_key": keys_dict["OpenAI"],
            "model": "gpt-6.1-sol",
            "providers": None,
            "quantization_tiers": None
        },
        "claude": {
            "class": OpenRouterApi,
            "api_key": keys_dict["OpenRouter"],
            "model": "anthropic/claude-sonnet-5-5",
            "providers": ["google-vertex/global"],
            "quantization_tiers": None
        },
        "llama": {
            "class": OpenRouterApi,
            "api_key": keys_dict["OpenRouter"],
            "model": "meta-llama/llama-4-maverick",
            "providers": ["parasail/fp8"],
            "quantization_tiers": [["bf16", "fp16"], ["fp8"]]
        },
        "mistral": {
            "class": OpenRouterApi,
            "api_key": keys_dict["OpenRouter"],
            "model": "mistralai/mistral-medium-3-5",
            "providers": ["mistral/zdr"],
            "quantization_tiers": None
        },
        "gemini": {
            "class": OpenRouterApi,
            "api_key": keys_dict["OpenRouter"],
            "model": "google/gemini-3.5-flash-lite",
            "providers": ["google-vertex/global"],
            "quantization_tiers": None
        },
        "deepseek": {
            "class": OpenRouterApi,
            "api_key": keys_dict["OpenRouter"],
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "providers": ["inference-net/fp8"],
            "quantization_tiers": [["bf16", "fp16"], ["fp8"]]
        },
    }

    log.info(f"Models to run: {model_configs}")
    
    if github_actions:
        model_configs[model_name]["model"] = model_version

    log.info(f"Models to run: {model_configs}")

    model_dict = {}
    for model in args["models"]:
        if model in model_configs:
            config = model_configs[model]
            model_dict[model] = config["class"](
                api_key=config["api_key"],
                model=config["model"],
                system_role=args["system_role"],
                temperature=args["temperature"],
                providers=config["providers"],
                quantization_tiers=config["quantization_tiers"],
            )

        else:
            print(f"Warning: Unknown model '{model}' requested. Skipping.")

    return model_dict


def LR_transition_values(args, data, model_name):
    """
    Performs per-quantity logistic regression to find transition points
    for the given model_name. Rows with missing reward_value or output_binary
    are dropped. The function merges results into 'results/<prompt_type>/<experiment_name>/LR_results.pkl',
    keyed by model_name.
    Args:
        args: Command-line or config arguments (not directly used here,
              included for a consistent signature).
        data (dict): A dictionary {model_name: DataFrame} returned from your main experiment,
                     from which we extract data[model_name].
        model_name (str): Identifier for the model (e.g. "gpt4o").
    Returns:
        None. Loads and updates 'LR_results.pkl' so that LR_results[model_name] = DataFrame.
    """
    df = data[model_name].copy()

    if args["prompt_type"] == "chinese":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if (
                    re.search(
                        r"^\s*是\s*(?:"
                        r"的"
                        r"|（.*?）"
                        r"|\(.*?\)"
                        r")?\s*[。.．!！?？]?\s*$",
                        str(x or "").strip(),
                    )
                    or re.search(r"\byes\b", str(x or "").strip(), flags=re.IGNORECASE)
                )
                else (
                    0
                    if (
                        re.search(r"^\s*否\s*[。.．!！?？]?\s*$", str(x or "").strip())
                        or re.search(
                            r"^\s*不(?:是)?\s*[。.．!！?？]?\s*$", str(x or "").strip()
                        )
                        or re.search(
                            r"\bno\b", str(x or "").strip(), flags=re.IGNORECASE
                        )
                    )
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "french":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"^\s*oui\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"^\s*non\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "dutch":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"^\s*ja\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"^\s*nee\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "cot":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if (
                    re.search(r"answer", str(x or ""), re.IGNORECASE)
                    and re.search(r"yes\b", str(x or ""), re.IGNORECASE)
                )
                else (
                    0
                    if (
                        re.search(r"answer", str(x or ""), re.IGNORECASE)
                        and re.search(r"no\b", str(x or ""), re.IGNORECASE)
                    )
                    else np.nan
                )
            )
        )
    else:
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"\byes\b", x.strip(), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"\bno\b", x.strip(), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )

    more_than_two_words = df["output"].str.split().str.len() > 2
    if more_than_two_words.any() and args["prompt_type"] != "cot":
        print(
            "[INFO] Found outputs with more than two words at these indices for: ",
            f"**{model_name}** model in prompt_type: **{args['prompt_type']}**: ",
        )
        for idx in df.index[more_than_two_words]:
            print(f"\t-index {idx}: '{df.loc[idx, 'output']}'")
        print(
            "[INFO] Consider manual checking of the classification of aforementioned index value in 'utils.LR_transition_values' function."
        )

    mask_nan = df["reward_value"].isna() | df["output_binary"].isna()
    df_nan = df[mask_nan]
    if not df_nan.empty:
        print(
            f'\nModel: **{model_name}**, prompt type: **{args["prompt_type"]}**\n',
            df_nan,
        )
        print(f"\n[INFO] Dropping {len(df_nan)} rows due to NaN.\n")
    df = df.dropna(subset=["reward_value", "output_binary"])

    unique_quantities = sorted(df["quantity"].unique())
    results = []

    for q in unique_quantities:
        sub = df[df["quantity"] == q]

        X_raw = sub[["reward_value"]].astype(float)
        if args["log_scale_money"]:
            if (X_raw <= 0).any().any():
                raise ValueError(
                    f"Found non‑positive reward_value(s) in quantity '{q}' while log_scale=True."
                )
            X = np.log10(X_raw)
        else:
            X = X_raw

        y = sub["output_binary"].astype(int)

        # Skip degenerate cases where y is constant
        if y.nunique() == 1:
            if args["log_scale_money"]:
                if y.iloc[0] == 0:
                    note_str = f'>{10**args["money_max"]}'
                else:
                    note_str = f'<{10**args["money_min"]}'
            else:
                if y.iloc[0] == 0:
                    note_str = f'>{args["money_max"]}'
                else:
                    note_str = f'<{args["money_min"]}'

            results.append((q, note_str))
            continue

        # Fit logistic regression
        model = LogisticRegression()
        model.fit(X, y)
        beta0 = model.intercept_[0]
        beta1 = model.coef_[0][0]

        if beta1 == 0:
            results.append((q, "Fit slope=0?"))
            continue

        # 50% acceptance => (beta0 + beta1*x)=0 => x*=-beta0/beta1
        x_star = -beta0 / beta1

        if args["log_scale_money"]:
            x_star = 10**x_star

        results.append((q, x_star))

    results_df = pd.DataFrame(results, columns=["quantity", "transition"])

    pickle_filename = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/LR_results.pkl'
    try:
        with open(pickle_filename, "rb") as f:
            LR_results = pickle.load(f)
    except FileNotFoundError:
        LR_results = {}

    LR_results[model_name] = results_df
    with open(pickle_filename, "wb") as f:
        pickle.dump(LR_results, f)


def LR_values_per_quantity_table(args):
    """
    Build a trade‑off table that contains only the (prompt_type, quantity)
    pairs listed under args['trade_off_values'][experiment_name].
    - Rows  = models
    - Cols  = '<prompt_type>|<requested_quantity>'
      (shows exact value if present, or the numerically closest one)
    """
    output_dir = args["output_dir"]
    experiment_name = args["experiment_name"]
    trade_cfg = args.get("trade_off_values", {})

    if "value" in trade_cfg:
        trade_cfg = trade_cfg["value"]

    if experiment_name not in trade_cfg:
        raise KeyError(
            f"[INFO] {experiment_name!r} not found in args['trade_off_values']."
        )

    cfg_quantities = trade_cfg[experiment_name]

    all_tables = []
    for prompt_type, wanted_list in cfg_quantities.items():
        pkl_path = f"{output_dir}/{prompt_type}/{experiment_name}/LR_results.pkl"
        if not os.path.exists(pkl_path):
            print(f"[INFO] {pkl_path} not found, skipping '{prompt_type}'.")
            continue

        with open(pkl_path, "rb") as f:
            LR_results = pickle.load(f)

        # gather available quantities once
        available_q = set()
        for df in LR_results.values():
            available_q.update(df["quantity"].unique())

        numeric_available = [
            float(q)
            for q in available_q
            if isinstance(q, (int, float)) or str(q).replace(".", "", 1).isdigit()
        ]

        q_use_map = {}  # {q_req: q_use}
        for q_req in wanted_list:
            if q_req in available_q:
                q_use_map[q_req] = q_req
            elif (
                isinstance(q_req, (int, float))
                or str(q_req).replace(".", "", 1).isdigit()
            ) and numeric_available:
                q_req_num = float(q_req)
                q_match = min(numeric_available, key=lambda x: abs(x - q_req_num))
                print(
                    f"[INFO] For {prompt_type} prompt: quantity {q_req} not found, "
                    f"using closest value: {q_match} for trade-offs table."
                )
                q_use_map[q_req] = q_match
            else:
                q_use_map[q_req] = None  # nothing suitable

        rows = []
        for model_name, df in LR_results.items():
            row = {"model_name": model_name}

            for q_req in wanted_list:
                q_use = q_use_map[q_req]
                col_name = (
                    f"{prompt_type}|{q_use}"
                    if q_use is not None
                    else f"{prompt_type}|{q_req}"
                )

                if q_use is None:
                    row[col_name] = None
                    continue

                t_val = df.loc[df["quantity"] == q_use, "transition"]
                row[col_name] = None if t_val.empty else t_val.iloc[0]

            rows.append(row)

        tdf = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
        all_tables.append(tdf)

    if not all_tables:
        raise RuntimeError(
            "[INFO] utils.LR_values_per_quantity_table: no LR_results.pkl files processed."
        )

    tradeoff_df = (
        reduce(
            lambda left, right: pd.merge(left, right, on="model_name", how="outer"),
            all_tables,
        )
        .sort_values("model_name")
        .reset_index(drop=True)
    )

    money_min = args["money_min"]
    money_max = args["money_max"]
    log_scale = args.get("log_scale_money", False)

    if log_scale:
        if isinstance(money_min, (int, float)) and isinstance(money_max, (int, float)):
            money_min, money_max = 10**money_min, 10**money_max

    def format_value(v):
        if isinstance(v, str) or pd.isna(v):
            return v
        if not isinstance(v, (int, float)):
            return v

        if v < money_min:
            return f"<{money_min}"
        elif v > money_max:
            return f">{money_max}"
        else:
            return round(v, 2)

    num_cols = tradeoff_df.columns.drop("model_name")
    for col in num_cols:
        tradeoff_df[col] = tradeoff_df[col].apply(format_value)

    global_table_path = f"{output_dir}/{experiment_name}_tradeoff_table.pkl"
    tradeoff_df.to_pickle(global_table_path)

    print("\nTrade‑off Table (requested quantities)")
    print(tradeoff_df.to_string(index=False))
    print(f"\nSaved to: {global_table_path}")


def retrieve_latest_model(model_name, api_key):

    if model_name == "deepseek":

        log.info(f"Retrieving latest model Open Router - Deepseek...")

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="deepseek",
                prefix="deepseek/",
                version_pattern=r"-v(\d+(?:\.\d+)*)",
                exclude=["batch"],
                max_input_per_m=5,
                max_output_per_m=15,
            )
        
        log.info(f"Latest model Open Router - Deepseek: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")

    elif model_name == "claude":

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="claude-sonnet",
                prefix="anthropic/",
                version_pattern=r"claude-sonnet-(\d+(?:\.\d+)*)",
                exclude=["batch"],
                max_input_per_m=5,
                max_output_per_m=15,
            )
        
        log.info(f"Latest model Open Router - Claude: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")


    elif model_name == "llama":

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="llama",
                prefix="meta-llama/",
                version_pattern=r"llama-(\d+(?:\.\d+)*)-",
                exclude=["batch"],
                max_input_per_m=5,
                max_output_per_m=15,
            )
        
        log.info(f"Latest model Open Router - Llama: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")


    elif model_name == "mistral":

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="mistral-medium",
                prefix="mistralai/",
                version_pattern=r"mistral-medium-(\d+(?:[.-]\d{1,2})?)",
                exclude=["batch"],
                max_input_per_m=5,
                max_output_per_m=15,
            )
        
        log.info(f"Latest model Open Router - Mistral: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")


    elif model_name == "gemini":

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="gemini",
                prefix="google/",
                version_pattern=r"gemini-(\d+(?:\.\d+)*)",
                exclude=["lite", "image", "tts", "audio"],
                max_input_per_m=5,
                max_output_per_m=15,
            )
        
        log.info(f"Latest model Open Router - Gemini: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")

    elif model_name == "gpt":

        model_version, price = retrieve_latest_model_open_router(
                api_key=api_key,
                query="gpt",
                prefix="openai/",
                version_pattern=r"gpt-(\d+(?:[.-]\d{1,2})?)-sol",
                exclude=["batch", "pro"],
                max_input_per_m=5,
                max_output_per_m=15,
            )

        model_version = model_version.split("openai/")[1]
        
        log.info(f"Latest model Open Router - GPT: {model_version}")
        log.info(f"Pricing per mio tokens: {price}")

    return model_version


def _per_million(value):
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def retrieve_latest_model_open_router(
        api_key, 
        query, 
        prefix, 
        version_pattern, 
        exclude=[], 
        max_input_per_m=None,
        max_output_per_m=None):
    
    version_re = re.compile(version_pattern)

    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        params={"sort": "newest", "q": query},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    candidates = []
    candidates_too_expensive = []

    for m in response.json()["data"]:
        model_id = m["id"]
        if not model_id.startswith(prefix) or model_id.endswith(":free"):
            continue

        if any(word in model_id.lower() for word in exclude):
            continue

        match = version_re.search(model_id)
        if not match:
            continue

        parts = [int(p) for p in re.split(r"[.-]", match.group(1))]
        version = tuple(parts + [0] * (3 - len(parts)))

        pricing = m.get("pricing") or {}
        prices = {
            "input_per_m": _per_million(pricing.get("prompt")),
            "output_per_m": _per_million(pricing.get("completion")),
        }

        if max_input_per_m is not None and (
            prices["input_per_m"] is None or prices["input_per_m"] > max_input_per_m
        ):
            candidates_too_expensive.append((model_id, prices))
            continue
        if max_output_per_m is not None and (
            prices["output_per_m"] is None or prices["output_per_m"] > max_output_per_m
        ):
            candidates_too_expensive.append((model_id, prices))
            continue

        candidates.append((version, model_id, prices))

    log.info(f"Candidates: {candidates}")
    log.info(f"Candidates too expensive: {candidates_too_expensive}")

    if not candidates:
        msg = f"No model found for query '{query}' within the price limits."
        if candidates_too_expensive:
            msg += f" Skipped (over budget or price unknown): {candidates_too_expensive}"
        raise RuntimeError(msg)

    _, model_id, prices = max(candidates, key=lambda c: c[0])

    return model_id, prices

"""
def heatmap_exists_for_current_month(api_key):

    api = HfApi(token=api_key)
    prefix = f"heatmap_{datetime.now(timezone.utc).strftime('%Y-%m')}-"

    log.info(f"Checking Hugging Face repo_id={REPO_ID} repo_type={REPO_TYPE} for prefix {prefix}")

    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    except RepositoryNotFoundError:
        return False

    matches = [f for f in files if f.startswith(prefix)]

    if matches:
        log.info(f"Found heatmap(s) for this month: {matches}")
    else:
        log.info(f"No heatmap found with prefix {prefix}")

    return bool(matches)


def push_image_to_huggingface(api_key, path):

    api = HfApi(token=api_key)

    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

    _ = api.upload_file(
        path_or_fileobj=path, 
        path_in_repo=f"heatmap_{stamp}.png",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update heatmap",
    )
"""
