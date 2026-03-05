# Cash or Comfort? How LLMs Value Your Inconvenience

Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, David Martens (2025)
University of Antwerp, Belgium

arXiv: [Cash or Comfort? How LLMs Value Your Inconvenience](https://arxiv.org/pdf/2506.17367)

## Abstract

Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behavior in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behavior of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger, and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., €1 to wait 10 hours), (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., €1,000 to wait 0 minutes), (5) models often make different decisions at rounded figures (e.g., €10, €100, or €1,000), and (6) smaller models fail to demonstrate consistent decision-making behavior. These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf.


## Repository Overview

| Path       | What it contains                                         |
| ---------- | -------------------------------------------------------- |
| `config/`  | A configuration file               |
| `figures/`    | Publication‑ready figures               |
| `notebooks/`    | Notebooks used in the analysis                |
| `results/`  | LLMs' responses               |
| `tradeoff_data/`  | Scripts to generate the LLM responses about trade-offs              |
| `tradeoff_heatmaps/`  | Scripts to make trade-offs heatmaps             |
| `tradeoff_table/`  | Scripts to make trade-offs table             |
| `utils/`  | Scripts containing the utils             |

## Quick Start

```bash
# Clone the repository
$ git clone https://github.com/ADMAntwerp/Cash-or-Comfort-How-LLMs-Value-Your-Inconvenience.git
$ cd LLM-tradeoffs

# (Optional) create a Python virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

4. How to use:

    We use a ```config/config_file.yaml``` to configurate the experiments. Repository consist of three modules: 
    * ```tradeoff_data``` - to generate the LLM's trade-offs data.
    * ```tradeoff_heatmaps``` - to generate the heatmaps from trade-offs data.
    * ```tradeoff_table``` - to generate the trade-offs table.

    To get the results for LLM's trade-offs specified in ```config/config_file.yaml```, run:
    ```
    python -m tradeoff_data
    ```

    To get the heatmaps from data, run:
    ```
    python -m tradeoff_heatmaps
    ```

    To get the trade-off table from data, run:
    ```
    python -m tradeoff_table
    ```


## Citation

If you use this repository or build on our work, please cite the article:

```bibtex
@article{cedro2025cash,
  title={Cash or Comfort? How LLMs Value Your Inconvenience},
  author={Cedro, Mateusz and Ichmoukhamedov, Timour and Goethals, Sofie and He, Yifan and Hinns, James and Martens, David},
  journal={arXiv preprint arXiv:2506.17367},
  year={2025}
}
```