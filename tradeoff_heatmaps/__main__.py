#!/usr/bin/env python3
"""
Main script to plot the results and tables for LLM Trade-Offs framework.
"""

from utils.utils import load_config
from tradeoff_heatmaps.heatmaps import make_tradeoff_heatmaps


def main():
    args = load_config()

    # Make Heatmaps
    make_tradeoff_heatmaps(args)


if __name__ == "__main__":
    main()
