import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


PROMPTS = [
    "time",
    "distance",
    "hunger",
    "pain",
]
MODEL_ORDER = [
    "llama3_3_70b",
    "gemini2",
    # "deepseek_v3",
    "mixtral8x22b",
    "gpt4o",
    "claude3_5",
]

MODEL_DISPLAY_NAMES = {
    "llama3_3_70b": "Llama 3.3 70B",
    "gemini2": "Gemini 2.0 Flash",
    "deepseek_v3": "DeepSeek V3",
    "mixtral8x22b": "Mixtral 8x22B",
    "gpt4o": "GPT-4o",
    "claude3_5": "Claude 3.5 Sonnet",
}

ROW_NAMES = {
    "time": "Time",
    "distance": "Distance",
    "hunger": "Hunger",
    "pain": "Pain",
}

ROW_SUBTITLES = {
    "time": "Appointment postponement [min]",
    "distance": "Distance to walk [km]",
    "hunger": "Delivery postponement [min]",
    "pain": "Pain signal [%]",
}


def compute_edges(centres, *, log=False):
    centres = np.asarray(centres)
    if centres.size == 1:
        c = centres[0]
        return (
            np.array([c / 10**0.5, c * 10**0.5])
            if log
            else np.array([c - 0.5, c + 0.5])
        )

    if not log:
        half = np.diff(centres) / 2
        e = np.empty(centres.size + 1)
        e[0] = centres[0] - half[0]
        e[-1] = centres[-1] + half[-1]
        e[1:-1] = centres[:-1] + half
        return e

    logc = np.log(centres)
    half = np.diff(logc) / 2
    le = np.empty(centres.size + 1)
    le[0] = logc[0] - half[0]
    le[-1] = logc[-1] + half[-1]
    le[1:-1] = logc[:-1] + half
    return np.exp(le)


def make_heat_map(df, model, prompt, plot_dir, *, log_y=True):
    df = df.copy()

    df["output_numeric"] = df["output"].map(
        lambda x: (
            1
            if re.search(r"\byes\b", x.strip(), flags=re.IGNORECASE)
            else 0 if re.search(r"\bno\b", x.strip(), flags=re.IGNORECASE) else np.nan
        )
    )

    reward = np.sort(df["reward_value"].unique())
    quant = np.sort(df["quantity"].unique())

    stack = []
    for _, g in df.groupby("experiment"):
        pv = (
            g.pivot(index="reward_value", columns="quantity", values="output_numeric")
            .reindex(index=reward, columns=quant)
            .fillna(0)
        )
        stack.append(pv.values)
    mean = np.mean(np.stack(stack, axis=0), axis=0)

    Xe = compute_edges(quant)
    Ye = compute_edges(reward, log=log_y)
    XX, YY = np.meshgrid(Xe, Ye)

    plt.figure(figsize=(3.2, 3.2))
    plt.pcolormesh(XX, YY, mean, shading="auto", cmap="viridis", vmin=0, vmax=1)

    ax = plt.gca()
    ax.set_box_aspect(1)

    plt.tight_layout(pad=1.2)

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    min_x = min(quant)
    max_x = max(quant)
    mid1 = min_x + (max_x - min_x) / 3
    mid2 = min_x + 2 * (max_x - min_x) / 3
    xticks = [min_x, mid1, mid2, max_x]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks])

    ax.tick_params(axis="both", which="major", labelsize=18)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.title("")

    os.makedirs(plot_dir, exist_ok=True)
    out_png = os.path.join(plot_dir, f"{model}_log.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return out_png


def assemble_grid(
    base_dir="figures/all", out_png="figures/all/combined_prompts_models.png"
):

    rows, cols = len(PROMPTS), len(MODEL_ORDER)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.0, rows * 3.5),
        squeeze=False,
    )

    plt.subplots_adjust(
        left=0.18,
        right=0.9,
        top=0.9,
        bottom=0.18,
        hspace=0.0,
        wspace=-0.005,
    )

    for r, prompt in enumerate(PROMPTS):
        for c, model in enumerate(MODEL_ORDER):
            ax = axs[r, c]
            png = os.path.join(base_dir, prompt, f"{model}_log.png")
            if os.path.isfile(png):
                ax.imshow(mpimg.imread(png))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            ax.set_box_aspect(1)

    for r, prompt in enumerate(PROMPTS):
        for c, model in enumerate(MODEL_ORDER):
            axs[r, c].set_title(
                MODEL_DISPLAY_NAMES.get(model, model),
                fontsize=12,
                pad=0,
                x=0.57,
                y=0.94,
            )

    for r, prompt in enumerate(PROMPTS):
        pos = axs[r, 0].get_position()
        fig.text(
            0.17,
            0.005 + pos.y0 + pos.height / 2,
            ROW_NAMES.get(prompt, prompt),
            ha="left",
            va="center",
            fontsize=13,
            rotation="vertical",
            weight="bold",
        )

    for r, prompt in enumerate(PROMPTS):
        pos = axs[r, 0].get_position()
        y_txt = pos.y0
        fig.text(
            0.55,
            y_txt + 0.01,
            ROW_SUBTITLES[prompt],
            ha="center",
            va="top",
            fontsize=13,
            weight="bold",
        )

    fig.text(
        0.14,
        0.55,
        "Reward Offered [in Euro, log scale]",
        rotation="vertical",
        va="center",
        fontsize=18,
        weight="bold",
    )

    cax = fig.add_axes([0.4, 0.16, 0.3, 0.015])

    sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(
        "Probability of Acceptance",
        fontsize=13,
    )
    cb.ax.tick_params(labelsize=11)

    fig.suptitle("LLM Trade-Offs ", fontsize=20, x=0.55, y=0.94, weight="bold")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[OK]  mega-grid saved →  {out_png}")


if __name__ == "__main__":

    for prompt in PROMPTS:
        pkl = f"results_temp0/time/rev_21x21x5_temp0/data_log.pkl"
        out_dir = f"results_temp0/time/rev_21x21x5_temp0/"

        with open(pkl, "rb") as fh:
            dct = pickle.load(fh)

        for m in MODEL_ORDER:
            make_heat_map(dct[m], model=m, prompt=prompt, plot_dir=out_dir, log_y=True)

    assemble_grid()
