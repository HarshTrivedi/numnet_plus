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
    prepare_beaker_image, safe_create_dataset,
    dataset_name_to_id, image_name_to_id
)

from train_on_beaker import make_beaker_experiment_name as make_train_beaker_experiment_name
from processing_scripts.lib import hash_object


def load_dataset_mounts(train_experiment_name: str, evaluation_filepath: str) -> List[Dict]:

    # Setup Model Mount
    beaker_dataset_mounts = [{
        "datasetId": "tushark/numnet_roberta",
        "containerPath": "/model",
    }]

    # Mount training experiment dir so that archive can be found
    experiment_details = subprocess.check_output([
        "beaker", "experiment", "inspect", "--format", "json", "harsh-trivedi/"+train_experiment_name
    ]).strip()
    experiment_details = json.loads(experiment_details)
    trained_model_dataset_id = experiment_details[0]["executions"][-1]["result"]["beaker"]
    beaker_dataset_mounts.append({"datasetId": trained_model_dataset_id, "containerPath": f"/ckpt/"})

    # Setup Dev/Test file mount
    dataset_name = safe_create_dataset(evaluation_filepath)
    dataset_id = dataset_name_to_id(dataset_name)
    file_name = os.path.basename(evaluation_filepath)
    beaker_dataset_mounts.append({
        "datasetId": dataset_id,
        "subPath": file_name,
        "containerPath": f"/input/drop_dataset_dev.json"
    })
    beaker_dataset_mounts.append({ # mount dev on train.jsonl because numnet code needs something there.
        "datasetId": dataset_id,
        "subPath": file_name,
        "containerPath": f"/input/drop_dataset_train.json"
    })

    return beaker_dataset_mounts


def make_beaker_experiment_name(experiment_name: str, evaluation_filepath: str) -> str:
    hash_ = hash_object(experiment_name+evaluation_filepath)[:10]
    experiment_name = "predict_numnetplusv2_" + experiment_name[:80] + "__" + hash_
    return experiment_name


def make_beaker_experiment_description(train_experiment_name: str, evaluation_filepath: str) -> str:
    experiment_description = (
        f"Numnetplusv2: Predict on {evaluation_filepath} based on trained model : {train_experiment_name}"
    )
    return experiment_description


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help="Experiment name.")
    parser.add_argument('evaluation_filepath', type=str, help='Evaluation file path relative to project root.')
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
    train_experiment_name = make_train_beaker_experiment_name(args.experiment_name)
    dataset_mounts = load_dataset_mounts(train_experiment_name, args.evaluation_filepath)

    image_prefix = "numnetplusv2"
    beaker_image = prepare_beaker_image(
        dockerfile="numnet_plus/Dockerfile",
        allow_rollback=args.allow_rollback,
        beaker_image_prefix=image_prefix
    )

    beaker_image_id = image_name_to_id(beaker_image)

    arguments = ["sh", "predict_beaker.sh"]

    # Prepare Experiment Config
    beaker_experiment_name = make_beaker_experiment_name(args.experiment_name, args.evaluation_filepath)
    beaker_experiment_description = make_beaker_experiment_description(args.experiment_name, args.evaluation_filepath)

    task_config = {
         "spec": {
             "description": beaker_experiment_description,
             "image": beaker_image_id,
             "resultPath": "/output",
             "args": arguments,
             "datasetMounts": dataset_mounts,
             "requirements": {"gpuCount": 1},
             "env": {
                 "DATA_DIR": "/input/",
                 "MODEL_DIR": "/model/",
                 "CKPT_DIR": "/ckpt/",
                 "OUT_DIR": "/output/",
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
