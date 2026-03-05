import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Configuration ---
PROMPTS = [
    "time",
    "temp=0",
    "cot",
]
MODEL_ORDER = [
    "llama3_3_70b",
    "gemini2",
    "deepseek_v3",
    "mixtral8x22b",
    "gpt4o",
    "claude3_5",
    # "llama3_2_1b",
    # "llama3_2_3b",
    # "llama3_1_8b",
    # "llama3_3_70b",
]

MODEL_DISPLAY_NAMES = {
    # "llama3_2_1b": "Llama 3.2 1B",
    # "llama3_2_3b": "Llama 3.2 3B",
    # "llama3_1_8b": "Llama 3.1 8B",
    "llama3_3_70b": "Llama 3.3 70B",
    "gemini2": "Gemini 2.0 Flash",
    "deepseek_v3": "DeepSeek V3",
    "mixtral8x22b": "Mixtral 8x22B",
    "gpt4o": "GPT-4o",
    "claude3_5": "Claude 3.5 Sonnet",
}

ROW_NAMES = {
    # "time": "Time",
    # "distance": "Distance",
    # "hunger": "Hunger",
    # "pain": "Pain",
    "time": "Time (temp=1.0)",
    "temp=0": "Time (temp=0.0)",
    "cot": "Time (CoT, temp=1.0)",
    # "distance_safe": "Safe Distance",
}

ROW_SUBTITLES = {
    # "time": "Appointment postponement [min]",
    # "distance": "Distance to walk [km]",
    # "hunger": "Delivery postponement [min]",
    # "pain": "Pain signal [%]",
    "cot": "Appointment postponement [min]",
    "time": "Appointment postponement [min]",
    "temp=0": "Appointment postponement [min]",
    # "distance_safe": "Distance to walk [km]",
}
# ---------------------


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


def make_heat_map(df, model, prompt, plot_dir=None, *, log_y=True, ax=None):
    """
    Draws a heatmap.
    If 'ax' is provided, draws onto that axis (for the combined grid).
    If 'ax' is None, creates a new figure and saves it locally (for individual plots).
    """
    df = df.copy()
    # print(df.head(2)) # Removed print for cleaner output

    if prompt != "cot":
        df["output_numeric"] = df["output"].map(
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

    # --- Determine target axis ---
    if ax is None:
        # We are making an individual plot
        fig, target_ax = plt.subplots(figsize=(3.2, 3.2))
    else:
        # We are drawing onto the main grid
        target_ax = ax

    # --- Plotting ---
    # FIX 1: Added edgecolor='face' and linewidth=0 to fix white lines within heatmap (Image 3)
    target_ax.pcolormesh(
        XX,
        YY,
        mean,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        # edgecolor="face",
        # linewidth=0.1,
        rasterized=True,
    )

    target_ax.set_box_aspect(1)

    if log_y:
        target_ax.set_yscale("log")
        target_ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        target_ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        # FIX 2: Removed the padding that caused the top gap (Image 4)
        # current_ylim = target_ax.get_ylim()
        # target_ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)
    else:
        target_ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    min_x = min(quant)
    max_x = max(quant)
    mid1 = min_x + (max_x - min_x) / 3
    mid2 = min_x + 2 * (max_x - min_x) / 3
    xticks = [min_x, mid1, mid2, max_x]
    target_ax.set_xticks(xticks)
    target_ax.set_xticklabels([f"{x:.0f}" for x in xticks])

    target_ax.tick_params(axis="both", which="major", labelsize=18, pad=1)

    # Ensure spines are thick and black for the "tight grid" look (Image 1)
    for spine in target_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    target_ax.set_xlabel("")
    target_ax.set_ylabel("")
    target_ax.set_title("")

    # --- Saving (Only if making individual plots) ---
    if ax is None and plot_dir is not None:
        plt.tight_layout(pad=1.2)
        os.makedirs(plot_dir, exist_ok=True)

        # Define filenames
        out_png = os.path.join(plot_dir, f"{model}_log.png")
        out_svg = os.path.join(plot_dir, f"{model}_log.svg")

        # Save individual formats
        plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)

        plt.close(fig)
        return out_png

def assemble_grid(all_data_dict, prompts_to_plot, out_base):
    rows, cols = len(prompts_to_plot), len(MODEL_ORDER)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.65, rows * 4.0), 
        squeeze=False,
    )

    # --- 1. Spacing between the grids ---
    # bottom margin is much smaller now so it doesn't waste space on a tall multi-row plot
    plt.subplots_adjust(
        left=0.10,   
        right=0.98,
        top=0.92,    
        bottom=0.10, 
        hspace=0.55, # <-- INCREASED to leave room for X-axis numbers and row subtitles
        wspace=0.2, 
    )

    for r, prompt in enumerate(prompts_to_plot):
        for c, model in enumerate(MODEL_ORDER):
            ax = axs[r, c]

            if prompt in all_data_dict and model in all_data_dict[prompt]:
                make_heat_map(
                    all_data_dict[prompt][model],
                    model=model,
                    prompt=prompt,
                    log_y=True,
                    ax=ax,
                )
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
                ax.set_box_aspect(1)

            # --- 2. Model names on top of the grids ---
            ax.set_title(
                MODEL_DISPLAY_NAMES.get(model, model),
                fontsize=17,
                pad=6, 
            )
            
            # Y-axis string label (e.g., "Time") for the first column only
            if c == 0:
                ax.set_ylabel(ROW_NAMES.get(prompt, prompt), fontsize=18, weight="bold")

    # --- 3. X label below EACH row ---
    for r, prompt in enumerate(prompts_to_plot):
        # Find the center point of the entire row for perfectly centered text
        pos_left = axs[r, 0].get_position()
        pos_right = axs[r, -1].get_position()
        center_x = (pos_left.x0 + pos_right.x1) / 2
        
        fig.text(
            center_x, 
            pos_left.y0 - 0.033, # Placed just below the X-axis tick numbers
            ROW_SUBTITLES[prompt],
            ha="center",
            va="top",
            fontsize=17,
            weight="bold",
        )

    # --- TOP TITLE ---
    fig.text(
        0.53, 
        0.97,  
        "Prompting Strategies", # "Inconvenience-Reward LLM Trade-Offs", 
        ha="center", 
        va="center",
        fontsize=24,
        weight="normal", 
    )

    # --- Main Y Label ---
    fig.text(
        0.03, # Pushed left to fit the new margin
        0.508,  # Centered vertically for the whole multi-row figure
        "Reward Offered", 
        rotation="vertical",
        va="center",
        ha="center", 
        fontsize=23,
        weight="bold",
    )

    # --- Main Y Label (Line 2: Smaller) ---
    fig.text(
        0.045, # Sits slightly closer to the heatmaps
        0.508,  # Centered vertically for the whole multi-row figure
        "[in Euro, log scale]", 
        rotation="vertical",
        va="center",
        ha="center", 
        fontsize=19,     
        weight="normal", 
    )

    # --- 4. Colorbar positioning ---
    # Positioned at the very bottom center
    cax = fig.add_axes([0.392, 0.012, 0.30, 0.02]) 

    sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(
        "Probability of Acceptance",
        fontsize=18,
        labelpad=5
    )
    cb.ax.tick_params(labelsize=15)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"mega-grid saved to: {out_base}.svg and .pdf")


if __name__ == "__main__":
    master_data = {}

    # --- NEW: Define exact paths for each condition ---
    DATA_PATHS = {
        "time": "results/time/rev_21x21x5_new_all/data_log.pkl",
        # "temp=0": "results/temp=0/time/rev_21x21x5_temp0/data_log.pkl",
        "temp=0": "results/time_variations/temp=0/rev_21x21x5_new_all/data_log.pkl",
        "cot": "results/cot/rev_21x21x5_new_all/data_log.pkl",
    }

    # Step 1: Gather data and make individual plots for ALL prompts
    for prompt in PROMPTS:
        # Fetch the specific path from the dictionary
        pkl = DATA_PATHS.get(prompt)
        
        if pkl is None:
            print(f"Warning: No file path defined for '{prompt}', skipping...")
            continue

        out_dir_individual = f"new_figures/time_cot/{prompt}/"

        try:
            with open(pkl, "rb") as fh:
                dct = pickle.load(fh)
            master_data[prompt] = dct

            print(f"--- Generating individual plots for prompt: {prompt} ---")
            for m in MODEL_ORDER:
                if m in dct:
                    make_heat_map(
                        dct[m],
                        model=m,
                        prompt=prompt,
                        plot_dir=out_dir_individual,
                        log_y=True,
                    )

        except FileNotFoundError:
            print(f"Warning: Could not find {pkl}, skipping...")

    # Step 2: Generate ONE mega combined vector grid using ALL loaded prompts
    print(f"\n--- Generating MEGA combined vector grid for all prompts ---")
    assemble_grid(
        all_data_dict=master_data,
        prompts_to_plot=PROMPTS, 
        out_base="new_figures/time_cot/combined_all_prompts", 
    )