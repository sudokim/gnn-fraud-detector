from argparse import ArgumentParser, Namespace

__all__ = ["parse_args"]


def _validate_args(args: Namespace):
    if args.gcn_num_layers < 2:
        raise ValueError("GCN should have at least 2 layers")


def parse_args():
    parser = ArgumentParser()

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset_mat_path", type=str, default="dataset/Amazon.mat", help="Path to the dataset in .mat format"
    )
    data.add_argument("--dataset_split_seed", type=int, default=42, help="Seed for splitting the dataset")

    module = parser.add_argument_group("Module")
    module.add_argument("--module_type", type=str, default="GCN", help="Type of the module to use")
    module.add_argument("--module_weight_decay", type=float, default=5e-4, help="Weight decay of the module")
    module.add_argument("--module_lr", type=float, default=0.01, help="Learning rate of the module")

    module_gcn = module.add_argument_group("GCN")
    module_gcn.add_argument("--gcn_hidden_dim", type=int, default=16, help="Hidden dimension of the GCN module")
    module_gcn.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout rate of the GCN module")
    module_gcn.add_argument("--gcn_num_layers", type=int, default=2, help="Number of layers of the GCN module")
    module_gcn.add_argument("--gcn_pos_weight", type=float, default=10, help="Positive class weight of the GCN module")

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--trainer_max_epochs", type=int, default=2000, help="Maximum number of epochs to train")
    trainer.add_argument("--seed", type=int, default=42, help="Seed for the trainer")

    args = parser.parse_args()

    _validate_args(args)

    return args


def parse_bert_args():
    parser = ArgumentParser()

    module = parser.add_argument_group("Module")
    module.add_argument("--model_name", type=str, default="bert-base-cased", help="Type of the model to use")
    module.add_argument("--module_weight_decay", type=float, default=0.01, help="Weight decay of the module")
    module.add_argument("--module_lr", type=float, default=5e-4, help="Learning rate of the module")

    trainer = parser.add_argument_group("Trainer")

    trainer.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    trainer.add_argument("--seed", type=int, default=42, help="Seed for the trainer")

    return parser.parse_args()