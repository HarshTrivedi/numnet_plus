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
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from beaker_scripts.util import (
    prepare_beaker_image,
    safe_create_dataset,
    dataset_name_to_id,
    image_name_to_id,
    beaker_name_from_training_experiment
)

from processing_scripts.lib import read_jsonl, write_jsonl, hash_object, clean_white_space, split_list
from cache_data_on_beaker import make_beaker_experiment_name as make_cache_data_beaker_experiment_name


def load_dataset_mounts(dev_filepath: str, cached_data_experiment_name: str, pretrain_experiment_name: str = None) -> List[Dict]:

    # Setup Model Mount
    beaker_dataset_mounts = [{
        "datasetId": "tushark/numnet_roberta",
        "containerPath": "/model",
    }]

    # Mount pretraining experiment dir so that archive can be found
    if pretrain_experiment_name:
        experiment_details = subprocess.check_output([
            "beaker", "experiment", "inspect", "--format", "json", "harsh-trivedi/"+pretrain_experiment_name
        ]).strip()
        experiment_details = json.loads(experiment_details)
        trained_model_dataset_id = experiment_details[0]["executions"][-1]["result"]["beaker"]
        beaker_dataset_mounts.append({"datasetId": trained_model_dataset_id, "containerPath": f"/ckpt/"})

    # Setup cached train and dev files
    try:
        experiment_details = subprocess.check_output([
            "beaker", "experiment", "inspect", "--format", "json", "harsh-trivedi/"+cached_data_experiment_name
        ]).strip()
    except:
        exit(f"Cached data experiment not found harsh-trivedi{cached_data_experiment_name}.")
    experiment_details = json.loads(experiment_details)
    cached_data_dataset_id = experiment_details[0]["executions"][-1]["result"]["beaker"]
    beaker_dataset_mounts.append({"datasetId": cached_data_dataset_id, "containerPath": f"/cache/"})

    # Setup Dev file mount. This is required for final evaluation.
    dataset_name = safe_create_dataset(dev_filepath)
    dataset_id = dataset_name_to_id(dataset_name)
    file_name = os.path.basename(dev_filepath)
    beaker_dataset_mounts.append({
        "datasetId": dataset_id,
        "subPath": file_name,
        "containerPath": f"/input/drop_dataset_dev.json"
    })

    return beaker_dataset_mounts


def make_beaker_experiment_name(experiment_name: str) -> str:
    hash_ = hash_object(experiment_name)[:10]
    experiment_name = "train_numnetplusv2_" + experiment_name[:80] + "__" + hash_
    return experiment_name


def make_beaker_experiment_description(experiment_name: str) -> str:
    experiment_description = f"Numnetplusv2: Train config {experiment_name}"
    return experiment_description


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help="Experiment name.")
    parser.add_argument('--pretrain_experiment_name', type=str, help="Pretrain experiment name.", default=None)
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

    random_seed = experiment_config.pop("random_seed", 345)
    epochs = experiment_config.pop("epochs", 5)
    num_instances_per_epoch = experiment_config.pop("num_instances_per_epoch", None)
    learning_rate = experiment_config.pop("learning_rate", 5e-4)
    bert_learning_rate = experiment_config.pop("bert_learning_rate", 1.5e-5)
    weight_decay = experiment_config.pop("weight_decay", 5e-5)
    bert_weight_decay = experiment_config.pop("bert_weight_decay", 0.01)
    batch_size = experiment_config.pop("batch_size", 16)
    gradient_accum = experiment_config.pop("gradient_accum", 8)
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
    pretrain_experiment_name = None
    if args.pretrain_experiment_name:
        pretrain_experiment_name = make_beaker_experiment_name(args.pretrain_experiment_name)

    assert skip_tagging in (True, False)
    cache_data_experiment_name = make_cache_data_beaker_experiment_name(train_filepath, dev_filepath, skip_tagging)
    dataset_mounts = load_dataset_mounts(dev_filepath, cache_data_experiment_name, pretrain_experiment_name)

    image_prefix = "numnetplusv2"
    beaker_image = prepare_beaker_image(
        dockerfile="numnet_plus/Dockerfile",
        allow_rollback=args.allow_rollback,
        beaker_image_prefix=image_prefix
    )

    beaker_image_id = image_name_to_id(beaker_image)


    arguments = [
        "sh", "train_beaker.sh",
        str(random_seed), str(epochs), str(num_instances_per_epoch), str(learning_rate), str(bert_learning_rate),
        str(weight_decay), str(bert_weight_decay), str(batch_size), str(gradient_accum), "tag_mspan"
    ]
    if skip_tagging:
        arguments.pop()

    # Prepare Experiment Config
    beaker_experiment_name = make_beaker_experiment_name(args.experiment_name)
    beaker_experiment_description = make_beaker_experiment_description(args.experiment_name)
    wandb_run_name = uuid.uuid4().hex

    task_config = {
         "spec": {
             "description": beaker_experiment_description,
             "image": beaker_image_id,
             "resultPath": "/output",
             "args": arguments,
             "datasetMounts": dataset_mounts,
             "requirements": {"gpuCount": 1},
             "env": {
                 "DATA_DIR": "/input",
                 "MODEL_DIR": "/model",
                 "CKPT_DIR": "/ckpt",
                 "OUT_DIR": "/output",
                 "CACHE_DIR": "/cache",
                 "WANDB_RUN_NAME": wandb_run_name,
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
