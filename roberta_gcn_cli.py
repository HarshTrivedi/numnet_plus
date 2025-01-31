import os
import json
import options
import argparse
import torch
from pprint import pprint
from tools.model import DropBertModel
from mspan_roberta_gcn.roberta_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from tag_mspan_robert_gcn.roberta_batch_gen_tmspan import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet
from datetime import datetime
from tools.utils import create_logger, set_environment
from pytorch_transformers import RobertaTokenizer, RobertaModel


parser = argparse.ArgumentParser("Bert training task.")
options.add_bert_args(parser)
options.add_model_args(parser)
options.add_data_args(parser)
options.add_train_args(parser)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump(vars(args), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps
logger = create_logger("Bert Drop Pretraining", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)


def maybe_track_wandb(project_name: str = "synth2realmh"):
    import wandb
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
    wandb_api_key_path = ".project-wandb-config.json"
    print("Trying to setup wandb...")
    if not os.path.exists(wandb_api_key_path):
        print("Wandb API key not found.")
    if not wandb_run_name:
        print("Wandb run_name not found.")
    if os.path.exists(wandb_api_key_path) and wandb_run_name is not None:
        with open(wandb_api_key_path, "r") as file:
            print("Starting wandb.")
            wandb_api_key = json.load(file)["wandb_api_key"]
            wandb.login(key=wandb_api_key)
            wandb.init(project=project_name, name=wandb_run_name, id=wandb_run_name)

def main():
    maybe_track_wandb()

    best_result = float("-inf")
    logger.info("Loading data...")
    if not args.tag_mspan:
        logger.info("Loading data without tag_mspan.")
        train_itr = DropBatchGen(args, data_mode="train", tokenizer=tokenizer, make_infinite=True, lazy=args.lazy)
        dev_itr = DropBatchGen(args, data_mode="dev", tokenizer=tokenizer)
    else:
        logger.info("Loading data with tag_mspan.")
        train_itr = TDropBatchGen(args, data_mode="train", tokenizer=tokenizer, make_infinite=True, lazy=args.lazy)
        dev_itr = TDropBatchGen(args, data_mode="dev", tokenizer=tokenizer)

    if args.num_instances_per_epoch in (None, "None", "none", "null", "Null"):
        args.num_instances_per_epoch = len(train_itr) * args.batch_size
    args.num_instances_per_epoch = int(args.num_instances_per_epoch)
    num_batches_per_epoch = int(int(args.num_instances_per_epoch) / args.batch_size)

    num_train_steps = int(args.max_epoch * num_batches_per_epoch / args.gradient_accumulation_steps)
    num_batch_steps = int(args.max_epoch * num_batches_per_epoch)
    logger.info("Total num update steps {}!".format(num_train_steps))
    logger.info("Total num batches      {}!".format(args.max_epoch * num_batches_per_epoch))
    logger.info("Total num instances    {}!".format(args.max_epoch * args.num_instances_per_epoch))

    logger.info("Build bert model.")
    bert_model = RobertaModel.from_pretrained(args.roberta_model)

    logger.info("Build Drop model.")
    if not args.tag_mspan:
        network = NumericallyAugmentedBertNet(bert_model,
                 hidden_size=bert_model.config.hidden_size,
                 dropout_prob=args.dropout,
                 use_gcn=args.use_gcn,
                 gcn_steps=args.gcn_steps)
    else:
        network = TNumericallyAugmentedBertNet(bert_model,
                                              hidden_size=bert_model.config.hidden_size,
                                              dropout_prob=args.dropout,
                                              use_gcn=args.use_gcn,
                                              gcn_steps=args.gcn_steps)

    if args.cuda:
        network.cuda()
    if args.pre_path:
        # To allow fine-tuning model starting from a given checkpoint.
        print("Load from pre path {}.".format(args.pre_path))
        network.load_state_dict(torch.load(args.pre_path))

    logger.info("Build optimizer etc...")
    model = DropBertModel(args, network, num_train_step=num_train_steps)

    train_start = datetime.now()

    train_iterator = iter(train_itr)
    for epoch in range(1, args.max_epoch + 1):
        model.avg_reset()
        logger.info('At epoch {}'.format(epoch))
        logger.info('Number of batches in this epoch: {}'.format(num_batches_per_epoch))
        for step in range(num_batches_per_epoch):
            batch = next(train_iterator)
            model.update(batch)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info("Updates[{0:6}] train loss[{1:.5f}] train em[{2:.5f}] f1[{3:.5f}] remaining[{4}]".format(
                    model.updates, model.train_loss.avg, model.em_avg.avg, model.f1_avg.avg,
                    str((datetime.now() - train_start) / (step + 1) * (num_batches_per_epoch - step - 1)).split('.')[0]))
                model.avg_reset()
        total_num, eval_loss, eval_em, eval_f1 = model.evaluate(dev_itr)
        logger.info(
            "Eval {} examples, result in epoch {}, eval loss {}, eval em {} eval f1 {}.".format(total_num, epoch,
                                                                                                eval_loss, eval_em,
                                                                                                eval_f1))

        if eval_f1 > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_result = eval_f1
            logger.info("Best eval F1 {} at epoch {}".format(best_result, epoch))

    logger.info("done training in {} seconds!".format((datetime.now() - train_start).seconds))

if __name__ == '__main__':
    main()
