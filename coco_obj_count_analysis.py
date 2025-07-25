import subprocess
from datetime import datetime
# 0 still running, 11 still running, 10 weights missing!!!
# List of model configurations

model_configs = [
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_0/checkpoint.pt.tar",
        "ms_which_encoder_layers": "0"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_1/checkpoint.pt.tar",
        "ms_which_encoder_layers": "1"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_2/checkpoint.pt.tar",
        "ms_which_encoder_layers": "2"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_3/checkpoint.pt.tar",
        "ms_which_encoder_layers": "3"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_4/checkpoint.pt.tar",
        "ms_which_encoder_layers": "4"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_5/checkpoint.pt.tar",
        "ms_which_encoder_layers": "5"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_6/checkpoint.pt.tar",
        "ms_which_encoder_layers": "6"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_7/checkpoint.pt.tar",
        "ms_which_encoder_layers": "7"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_8/checkpoint.pt.tar",
        "ms_which_encoder_layers": "8"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_9/checkpoint.pt.tar",
        "ms_which_encoder_layers": "9"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_weights/dinosaur_11/checkpoint.pt.tar",
        "ms_which_encoder_layers": "11"
    }
]

# Common arguments for all evaluations
common_args = [
    "--dataset", "coco",
    "--data_path", "/fastdata/vilab03/COCO2017_some",  
    "--num_slots", "7",
    "--init_method", "shared_gaussian",
    "--eval_permutations", "standard",
    "--train_permutations", "standard",
    "--use_second_encoder", "False",
    "--concat_method", "none",
    "--seed", "1234321",
    "--encoder_final_norm", "False",
    "--log_path", "/visinf/home/vilab03/iccv_logs/coco_obj_count_analysis",
    "--visualize_attn", "True"
]


# Output log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"results_coco_some.txt"

# Evaluate each model and log results
with open(results_file, "w") as f:
    for i, config in enumerate(model_configs):
        cmd = ["python", "eval_spot.py"] + common_args + [
            "--checkpoint_path", config["checkpoint_path"],
            "--ms_which_encoder_layers", config["ms_which_encoder_layers"]
        ]
        f.write(f"\n\n### MODEL {config['ms_which_encoder_layers']}: {config['checkpoint_path']}, LAYERS: {config['ms_which_encoder_layers']} ###\n")
        f.write("Command: " + " ".join(cmd) + "\n")

        print(f"Running model {i+1}...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        f.write(result.stdout)
        f.write("\n" + "="*100 + "\n")

print(f"All evaluations completed. Results saved to: {results_file}")
