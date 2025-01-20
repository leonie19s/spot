import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# Constant of size guidance: When reading tensorboard event files, we only want to read all scalars
SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 1,
    event_accumulator.IMAGES: 1,
    event_accumulator.AUDIO: 1,
    event_accumulator.HISTOGRAMS: 1,
    event_accumulator.TENSORS: 1,
    event_accumulator.SCALARS: 0
}

# Some pretty colors, with assignment to proper fusion methods
COLORS = {
    "baseline": "#7F7F91",  # grey
    "residual": "#FFA600",  # orange
    "mean": "#3D1D9C",  # blue
    "max": "#C7007C",    # violet
    "sum": "#FF4647" # salmon
}

# Flag whether the (slots) should be removed from labels in plots
REMOVE_SLOTS_FROM_LABELS = True

# Constant base path to logging directory and plot path
BASE_PATH = "logs/"
PLOT_PATH = "plots/"


def results_for_run(run_name):

    # Get contents in that directory
    p_log_dir = os.path.join(BASE_PATH, run_name)
    log_dir_content = os.listdir(p_log_dir)

    # If the last element is a directory again, the timestamp of log path was not removed
    if os.path.isdir(os.path.join(p_log_dir, log_dir_content[-1])):
        print(f"Found subdirectories for log path {p_log_dir} - taking last subdirectory (last timestamp)")
        p_log_dir = os.path.join(p_log_dir, log_dir_content[-1])
        log_dir_content = os.listdir(p_log_dir)

    # Find tensorboard file
    tb_file_name_list = [x for x in log_dir_content if x.startswith("events.out.tfevents")]
    assert len(tb_file_name_list) == 1, "The provided directory must contain exactly one tensorboard event file"
    tb_file_name = tb_file_name_list[0]

    # Open the file and read the scalars for the validation
    p_tb_file = os.path.join(p_log_dir, tb_file_name)
    ev_acc = event_accumulator.EventAccumulator(p_tb_file, SIZE_GUIDANCE)
    ev_acc.Reload()
    scalars = ev_acc.Tags()["scalars"]
    val_scalars = [s for s in scalars if s.startswith("VAL")]

    # Create one joint pd.DataFrame from it
    scalar_df = pd.DataFrame()
    scalar_df.index.name = "Epoch"

    for scalar in val_scalars:
        scalar_read = ev_acc.Scalars(scalar)
        scalar_vals = [e.value for e in scalar_read]
        scalar_df[scalar[4:]] = scalar_vals     # Scalar starting from 4: to remove VAL/
    
    # Delete from memory to avoid OOM, then return
    del ev_acc
    return scalar_df


def get_color_for_run(run):
    for fusion, col in COLORS.items():
        if fusion in run:
            return col
    raise Exception(f"No respective color for run {run} could be found, must include fusion method")


def make_metrics_pretty(metrics):
    if REMOVE_SLOTS_FROM_LABELS:
        metrics = [m.replace(" (slots)", "") for m in metrics]
    return [r"${" + m.replace("mbo", "mBO").replace("miou", "mIoU") + "}$" for m in metrics]


def plot_one_metric(runs, run_dfs, pretty_labels, metric = "miou (slots)"):
    """
        Plots one metric over time.
    """

    # Set modern style
    plt.style.use("ggplot")

    # Create the plot
    fig, ax = plt.subplots(1, 1)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5, zorder=-1)

    # Plot the results
    for idx, run in enumerate(runs):
        label = run.split("/")[-1] if len(pretty_labels) == 0 else pretty_labels[idx]
        ax.plot(run_dfs[run].index , run_dfs[run][metric], label=label, color=get_color_for_run(run))

    # Aesthetics and saving
    plt.xlabel("Epochs")
    plt.ylabel(make_metrics_pretty([metric])[0])
    plt.legend()
    plt.savefig(os.path.join(PLOT_PATH, f"comparison_{metric}"))


def plot_comparison_in_multiple_metrics(
    runs, run_dfs, pretty_labels, metrics = np.array(["mbo_c (slots)", "mbo_i (slots)", "miou (slots)"])
):
    # Set modern style
    plt.style.use("ggplot")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid(True, linestyle='--', linewidth=0.9, alpha=0.75, zorder=-1)

    # Create the ticks with the metrics as labels
    diff_between_metrics = int(len(runs) * 2)
    tick_lst = [diff_between_metrics * x for x in range(len(metrics))]
    tick_label_lst = make_metrics_pretty(metrics)
    ax.set_xticks(tick_lst, labels=tick_label_lst)

    # Create offset list for aligning
    offsets = list(range(-int(len(runs)/2), int((len(runs) + 1)/2)))
    if len(runs) % 2 == 0:
        offsets[int(len(runs)/2):] = [x+1 for x in offsets[int(len(runs)/2):]]
        offsets = [x + 0.5 if x < 0 else x - 0.5 for x in offsets]

    # Create the bars with respective colors, labels and values
    for idx in range(len(run_dfs)):
        run = runs[idx]
        data = run_dfs[run].iloc[-1][metrics].values
        label = run.split("/")[-1] if len(pretty_labels) == 0 else pretty_labels[idx]
        ax.bar(np.array(list(range(len(metrics)))) * diff_between_metrics + offsets[idx], data, color=get_color_for_run(run), label=label, zorder=2, edgecolor="black") 

    # Aesthetics and saving
    ax.set_ylabel("Score")
    ax.set_ylim(30, 60)
    plt.legend()
    plt.savefig(os.path.join(PLOT_PATH, f"comparison_all_metrics"), bbox_inches="tight")


def main():

    # INSERT ALL RELEVANT RUNS FOR PLOTITNG HERE

    runs = [
        "baseline/dinosaur_baseline",
        "ablations/mean_9_10_11",
        "ablations/residual_5_8_11",
        "ablations/sum_9_10_11",
        "ablations/max_9_10_11"
    ]

    # Optional: Pretty labels, name them corresponding runs above. If none are to be used: leave as empty list
    pretty_labels = [
        "Baseline",
        "Mean",
        "Residual",
        "Sum",
        "Max"
    ]

    # runs = ["baseline/dinosaur_baseline", "ablations/residual_2_5_8_11"]
    # pretty_labels = ["Baseline", "Our best"]

    # Sanity check
    assert len(runs) == len(pretty_labels) or len(pretty_labels) == 0, "Either supply as many labels as there are runs, or none at all"

    # Reads in all dfs and stores them in a dictionary
    run_dfs = {}
    for run in runs:
        run_dfs[run] = results_for_run(run)

    # Basic plot for one metric

    # WHICH METRIC IS TO BE PLOTTED
    # Possible metrics: 
    # 'mse', 'ari (slots)', 'ari (decoder)', 'mbo_c', 'mbo_i', 'miou', 
    # 'mbo_c (slots)', 'mbo_i (slots)', 'miou (slots)', 'best_loss'
    plot_one_metric(runs, run_dfs, pretty_labels, "mbo_c (slots)")

    # Plot comparison in all metrics
    plot_comparison_in_multiple_metrics(runs, run_dfs, pretty_labels, np.array(["miou (slots)"]))


if __name__ == "__main__":
    main()