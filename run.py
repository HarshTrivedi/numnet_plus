"""
A script to dockerize a script for parallel processing of it on beaker.
"""
import math
import json
import argparse
import subprocess
import os
import sys
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from beaker_scripts.util import (
    prepare_beaker_image,
    safe_create_dataset,
    dataset_name_to_id,
    image_name_to_id,
    beaker_name_from_training_experiment
)

from processing_scripts.lib import read_jsonl, write_jsonl, hash_object, clean_white_space, split_list


def load_dataset_mounts(
        input_file_paths: List[str],
        working_dir="/run",
    ) -> List[Dict]:

    beaker_dataset_mounts = []
    for input_file_path in input_file_paths:

        input_file_time = datetime.utcfromtimestamp(os.path.getmtime(input_file_path)).replace(tzinfo=None)

        for index, split_file_path in enumerate(split_file_paths):

            dataset_name = safe_create_dataset(input_file_path)
            dataset_id = dataset_name_to_id(dataset_name)
            file_name = os.path.basename(input_file_path)

            beaker_dataset_mounts.append({
                "datasetId": dataset_id,
                "subPath": file_name,
                "containerPath": f"{working_dir}/{input_file_path}"
            })

    return beaker_dataset_mounts


def make_beaker_experiment_name(
        train_filepath: str, dev_filepath: str
    ) -> str:
    hash_object = hash_object(train_filepath+dev_filepath)[:10]
    experiment_name = "train_numnetplusv2_" + hash_object
    return experiment_name


def make_beaker_experiment_description(
        train_filepath: str, dev_filepath: str
    ) -> str:
    experiment_description = (
        f"Numnetplusv2: Train on {train_filepath}. Dev on {dev_filepath}"
    )
    return experiment_description


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help="Experiment name.")
    parser.add_argument(
        '--cluster', type=str,
        choices={"v100", "onperm-aristo", "onperm-ai2", "onperm-mosaic", "cpu"},
        default="v100"
    )
    parser.add_argument(
        '--dry_run', action='store_true', default=False,
        help='If specified, an experiment will not be created.'
    )
    parser.add_argument(
        '--allow_rollback', action='store_true', default=False,
        help='Allow rollback / use latest already present image.'
    )

    args = parser.parse_args()

    experiment_config_path = os.path.join(
        "experiment_configs", args.experiment_name + ".json"
    )
    with open(experiment_config_path, "r") as file:
        experiment_config = json.load(file)

    train_filepath = experiment_config["train_filepath"]
    dev_filepath = experiment_config["dev_filepath"]

    cluster_map = {
        "v100": "ai2/harsh-v100",
        "onperm-aristo": "ai2/aristo-cirrascale",
        "onperm-ai2": "ai2/general-cirrascale",
        "onperm-mosaic": "ai2/mosaic-cirrascale",
        "cpu": "ai2/harsh-cpu32",
    }
    cluster = cluster_map[args.cluster]

    CONFIGS_FILEPATH = ".project-beaker-config.json"
    with open(CONFIGS_FILEPATH) as file:
        configs = json.load(file)

    working_dir = configs.pop("working_dir")
    beaker_workspace = configs.pop("beaker_workspace")

    # Prepare Dataset Mounts
    dataset_mounts = load_dataset_mounts(train_filepath, dev_filepath)

    image_prefix = "numnetplusv2_"
    beaker_image = prepare_beaker_image(
        allow_rollback=args.allow_rollback,
        beaker_image_prefix=image_prefix
    )

    beaker_image_id = image_name_to_id(beaker_image)

    results_path = os.path.join(working_dir, output_directory)

    # Prepare Experiment Config

    beaker_experiment_name = make_beaker_experiment_name(train_filepath, dev_filepath)
    beaker_experiment_description = make_beaker_experiment_description(train_filepath, dev_filepath)


    arguments = [
        "sh", "train_beaker.sh", "345", "5e-4", "1.5e-5", "5e-5", "0.01", "16", "8"
    ]
    task_config = {
         "spec": {
             "description": beaker_experiment_description,
             "image": beaker_image_id,
             "resultPath": results_path,
             "args": arguments,
             "datasetMounts": task_dataset_mounts,
             "requirements": {"gpuCount": int(add_gpu)},
             "env": {}
         },
         "name": beaker_experiment_name,
         "cluster": cluster        
    }

    experiment_config = {"description": beaker_experiment_description, "tasks": [task_config]}

    # Save full config file.
    experiment_hash_id = hash_object(experiment_config)[:10]
    beaker_experiment_config_path = f".beaker_experiment_specs/{experiment_hash_id}.json"
    with open(beaker_experiment_config_path, "w") as output:
        output.write(json.dumps(experiment_config, indent=4))
    print(f"Beaker spec written to {beaker_experiment_config_path}.")

    # Build beaker command to run.
    experiment_run_command = [
        "beaker", "experiment", "create", beaker_experiment_config_path,
        "--name", beaker_experiment_name,
        "--workspace", beaker_workspace
    ]
    print(f"\nRun the experiment with:")
    print(f"    " + " ".join(experiment_run_command))

    # Run beaker command if required.
    if not args.dry_run:
        subprocess.run(experiment_run_command)

if __name__ == '__main__':
    main()
