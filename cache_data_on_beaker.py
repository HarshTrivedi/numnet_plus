"""
A script to dockerize a script for parallel processing of it on beaker.
"""
import math
import json
import _jsonnet
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

from processing_scripts.lib import read_jsonl, write_jsonl, hash_object, split_list


def load_dataset_mounts(train_filepath: str, dev_filepath: str) -> List[Dict]:

    # Setup Model Mount
    beaker_dataset_mounts = [{
        "datasetId": "tushark/numnet_roberta",
        "containerPath": "/model",
    }]

    # Setup Train file mount
    dataset_name = safe_create_dataset(train_filepath)
    dataset_id = dataset_name_to_id(dataset_name)
    file_name = os.path.basename(train_filepath)
    beaker_dataset_mounts.append({
        "datasetId": dataset_id,
        "subPath": file_name,
        "containerPath": f"/input/drop_dataset_train.json"
    })

    # Setup Dev file mount
    dataset_name = safe_create_dataset(dev_filepath)
    dataset_id = dataset_name_to_id(dataset_name)
    file_name = os.path.basename(dev_filepath)
    beaker_dataset_mounts.append({
        "datasetId": dataset_id,
        "subPath": file_name,
        "containerPath": f"/input/drop_dataset_dev.json"
    })

    return beaker_dataset_mounts


def make_beaker_experiment_name(train_filepath: str, dev_filepath: str, skip_tagging: bool, skip_train: bool = False) -> str:
    if not skip_train:
        hash_ = hash_object(train_filepath + dev_filepath + str(skip_tagging))[:10]
        train_filepath = train_filepath.replace("train", "train_or_dev")
        experiment_name = "cache_data_numnetplusv2_" + os.path.basename(train_filepath)[:50] + "__" + hash_
    else:
        hash_ = hash_object(dev_filepath + str(skip_tagging))[:10]
        experiment_name = "cache_data_numnetplusv2_" + os.path.basename(dev_filepath)[:50] + "__" + hash_
    return experiment_name


def make_beaker_experiment_description(train_filepath: str, dev_filepath: str, skip_tagging) -> str:
    experiment_description = f"Numnetplusv2: Cache train at: {train_filepath} and dev at: {dev_filepath}."
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
        "numnet_plus", "experiment_configs", args.experiment_name + ".jsonnet"
    )
    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_path))

    train_filepath = experiment_config.pop("train_filepath")
    dev_filepath = experiment_config.pop("dev_filepath")

    experiment_config.pop("random_seed", None)
    experiment_config.pop("epochs", None)
    experiment_config.pop("num_instances_per_epoch", None)
    experiment_config.pop("learning_rate", None)
    experiment_config.pop("bert_learning_rate", None)
    experiment_config.pop("weight_decay", None)
    experiment_config.pop("bert_weight_decay", None)
    experiment_config.pop("batch_size", None)
    experiment_config.pop("gradient_accum", None)
    skip_tagging = experiment_config.pop("skip_tagging", False)

    if experiment_config:
        exit(f"Some keys in experiment_config are not used: {experiment_config.keys()}")

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

    beaker_workspace = configs.pop("beaker_workspace")

    # Prepare Dataset Mounts
    dataset_mounts = load_dataset_mounts(train_filepath, dev_filepath)

    image_prefix = "numnetplusv2"
    beaker_image = prepare_beaker_image(
        dockerfile="numnet_plus/Dockerfile",
        allow_rollback=args.allow_rollback,
        beaker_image_prefix=image_prefix
    )

    beaker_image_id = image_name_to_id(beaker_image)

    arguments = ["sh", "cache_data_beaker.sh",  "tag_mspan"]
    if skip_tagging:
        arguments.pop()

    # Prepare Experiment Config
    beaker_experiment_name = make_beaker_experiment_name(train_filepath, dev_filepath, skip_tagging)
    beaker_experiment_description = make_beaker_experiment_description(train_filepath, dev_filepath, skip_tagging)

    task_config = {
         "spec": {
             "description": beaker_experiment_description,
             "image": beaker_image_id,
             "resultPath": "/output",
             "args": arguments,
             "datasetMounts": dataset_mounts,
             "requirements": {},
             "env": {
                 "DATA_DIR": "/input",
                 "MODEL_DIR": "/model",
                 "OUT_DIR": "/output",
             }
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
