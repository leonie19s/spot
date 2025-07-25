import subprocess
from datetime import datetime
# 0 still running, 11 still running, 10 weights missing!!!
# List of model configurations

model_configs = [
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_0_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "0"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_1_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "1"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_2_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "2"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_3_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "3"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_4_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "4"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_5_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "5"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_6_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "6"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_7_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "7"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_8_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "8"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_9_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "9"
    },
    {
        "checkpoint_path": "/fastdata/vilab01/single_layer_voc/layer_10_dinosaur/checkpoint.pt.tar",
        "ms_which_encoder_layers": "10"
    },
    {
        "checkpoint_path": "iccv_logs/ablations/abl_dinosaur_baseline/2025-02-27T19:22:07.341268/checkpoint.pt.tar",
        "ms_which_encoder_layers": "11"
    },
    {
        "checkpoint_path": "iccv_logs/main_results/voc/mr_voc_spot_studentsasignal_higherce_mfusion/2025-02-20T11:34:42.558533/checkpoint.pt.tar",
        "ms_which_encoder_layers": "8,9,10,11"
    },
    {
        "checkpoint_path": "iccv_logs/main_results/voc/spot_weights/checkpoint.pt.tar",
        "ms_which_encoder_layers": "11"
    }
]

# Common arguments for all evaluations
common_args = [
    "--dataset", "voc",
    "--data_path", "/fastdata/vilab03/VOCdevkit/VOC2012",  
    "--num_slots", "6",
    "--init_method", "shared_gaussian",
    "--eval_permutations", "standard",
    "--train_permutations", "standard",
    "--use_second_encoder", "False",
    "--concat_method", "none",
    "--seed", "1234321",
    "--encoder_final_norm", "False",
    "--log_path", "/visinf/home/vilab03/iccv_logs/coco_obj_count_analysis"
]

# VOC splits
voc_splits = ["single_val", "few_val", "some_val", "many_val"]

# Output log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"voc_obj_count_analysis_validation.txt"

# Evaluate each model and log results
with open(results_file, "w") as f:
    for voc_split in voc_splits:
        f.write(f"\n ############## \n   VOC SPLIT: {voc_split} OBJECT(S) \n ############## \n")
        for i, config in enumerate(model_configs):
            cmd = ["python", "voc_object_count_analysis.py"] + common_args + [
                "--checkpoint_path", config["checkpoint_path"],
                "--ms_which_encoder_layers", config["ms_which_encoder_layers"],
                "--voc_split", voc_split
            ]

            if (i > 11):
                for abc in range(len(cmd)):
                    if cmd[abc] == "shared_gaussian":
                        cmd[abc] = "embedding"
                    if cmd[abc] == "--eval_permutations":
                        cmd[abc+1] = "all"
                    if cmd[abc] == "--train_permutations":
                        cmd[abc+1] = "random"
                    if cmd[abc] == "--use_second_encoder":
                        cmd[abc+1] = "True"
                    if cmd[abc] == "--concat_method":
                        cmd[abc+1] = "denseconnector"

            if (i == 11 or i == 13):
                cmd.append("--use_original_spot_model")
                cmd.append("True")
                
            f.write(f"\n\n### MODEL {config['ms_which_encoder_layers']}: {config['checkpoint_path']}, LAYERS: {config['ms_which_encoder_layers']} ###\n")
            f.write("Command: " + " ".join(cmd) + "\n")

            print(f"Running model {i+1}...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            f.write(result.stdout)
            f.write("\n" + "="*100 + "\n")

print(f"All evaluations completed. Results saved to: {results_file}")
