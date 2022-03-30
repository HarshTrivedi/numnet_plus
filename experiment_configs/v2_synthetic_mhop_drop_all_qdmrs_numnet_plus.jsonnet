local qdmr_step_type_str = "all";
{
    "train_filepath": "processed_data/synthetic_data/sampled_for_mtl/v2_synthetic_mh_drop_" + qdmr_step_type_str + "_train_drop_format.jsonl",
    "dev_filepath": "processed_data/synthetic_data/sampled_for_mtl/v2_synthetic_mh_drop_" + qdmr_step_type_str + "_dev_drop_format.jsonl",
    "num_instances_per_epoch": 100000,
    "epochs": 20,
}
