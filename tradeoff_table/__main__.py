#!/usr/bin/env python3
"""
Main script to make a Trade-Off table.
"""

from utils.utils import load_config
from tradeoff_table.tradeoff_table import tradeoff_table


def main():
    args = load_config()

    # Make Trade-Off table
    tradeoff_table(args)


if __name__ == "__main__":
    main()
