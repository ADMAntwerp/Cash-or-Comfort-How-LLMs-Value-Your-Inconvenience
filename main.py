import logging
import argparse
import pickle
import yaml
import os
import json
import sys
from pathlib import Path
from datetime import date
from tradeoff_data.tradeoff_data import tradeoff_data
from utils.utils import retrieve_latest_model, load_config
from tradeoff_heatmaps.heatmaps import make_tradeoff_heatmaps


api_key_open_router = os.environ.get("API_KEY_OPEN_ROUTER")
if not api_key_open_router:
    raise RuntimeError("API_KEY_OPEN_ROUTER is not set")


github_actions = True


api_key_huggingface = os.environ.get("API_KEY_HUGGINGFACE")
if not api_key_huggingface:
    raise RuntimeError("API_KEY_HUGGINGFACE is not set")


"""  
github_actions = os.environ.get("GITHUB_ACTIONS")
if not github_actions:
    raise RuntimeError("GITHUB_ACTIONS is not set")
"""

    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger(__name__)


def load_state(path):
    if not path:
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_state(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def run_model(name, cfg):
    log.info("Running model %s", name)
    return [cfg["args"]["models"]["value"], date.today().strftime("%Y-%m")]


def check_if_run_is_needed():
    log.info("check_if_run_is_needed() started")
    
    # run_is_needed = not heatmap_exists_for_current_month(api_key_huggingface)
    run_is_needed = True

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"run_is_needed={'true' if run_is_needed else 'false'}\n")


def run():
    log.info("main() started")

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yml")
    p.add_argument("--model")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--final", action="store_true")
    ci_args = p.parse_args(sys.argv[2:])

    args = load_config()

    log.info(f"This job runs model: {ci_args.model}")

    state = load_state(ci_args.input)
    log.info("Loaded state with models: %s", list(state))

    if ci_args.final:
        make_tradeoff_heatmaps(args, state=state)
        push_image_to_huggingface(api_key_huggingface)
        log.info("Final job completed.")
        return

    log.info(ci_args.model)

    model_version = retrieve_latest_model(ci_args.model, api_key_open_router)

    result = tradeoff_data(args, model=ci_args.model, model_version=model_version, github_actions=github_actions)

    state.update(result)

    save_state(state, ci_args.output)
    log.info("Saved state with models: %s", list(state))


if __name__ == "__main__":
    command = sys.argv[1]
    {"check": check_if_run_is_needed, "run": run}[command]()
