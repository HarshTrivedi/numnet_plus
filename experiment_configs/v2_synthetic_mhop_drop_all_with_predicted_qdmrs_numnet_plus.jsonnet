local qdmr_step_type_str = "all_with_predicted";
{
    "train_filepath": "processed_data/synthetic_data/sampled_for_mtl/v2_synthetic_mh_drop_" + qdmr_step_type_str + "_train_drop_format.json",
    "dev_filepath": "processed_data/synthetic_data/sampled_for_mtl/v2_synthetic_mh_drop_" + qdmr_step_type_str + "_dev_drop_format.json",
    "num_instances_per_epoch": 100000,
    "epochs": 20,
}
