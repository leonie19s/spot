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


def main():

    # INSERT ALL RELEVANT RUNS FOR PLOTITNG HERE
    runs = [
        "baseline/dinosaur_baseline",
        "ablations/mean_9_10_11",
        "ablations/residual_5_8_11",
        "ablations/sum_9_10_11",
        "ablations/max_9_10_11"
    ]

    # WHICH METRIC IS TO BE PLOTTED
    # Possible metrics: 
    # 'mse', 'ari (slots)', 'ari (decoder)', 'mbo_c', 'mbo_i', 'miou', 
    # 'mbo_c (slots)', 'mbo_i (slots)', 'miou (slots)', 'best_loss'
    metric = "miou (slots)"

    # Reads in all dfs and stores them in a dictionary
    run_dfs = {}
    for run in runs:
        run_dfs[run] = results_for_run(run)

    # Create the plot
    fig, ax = plt.subplots(1, 1)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5, zorder=-1)

    # Plot the results
    for run in runs:
        ax.plot(run_dfs[run].index , run_dfs[run][metric], label=run.split("/")[-1])

    # Aesthetics
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(PLOT_PATH, metric))

    return


if __name__ == "__main__":
    main()