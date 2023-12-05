from argparse import ArgumentParser, Namespace

__all__ = ["parse_args"]


def _validate_args(args: Namespace):
    if args.gcn_num_layers < 2:
        raise ValueError("GCN should have at least 2 layers")


def parse_args():
    parser = ArgumentParser()

    data = parser.add_argument_group("Data")
    data_path = data.add_argument_group("Path")
    data_path.add_argument("--edge_index_path", type=str, default="dataset/amazon_instruments_user_all.npz")
    data_path.add_argument("--features_path", type=str, default="dataset/amazon_instruments_features.npy")
    data_path.add_argument("--labels_path", type=str, default="dataset/amazon_instruments_labels.pkl")
    data_path.add_argument("--split_file", type=str, default="dataset/amazon_instruments_split_masks.pkl")
    data_path.add_argument("--user_text_path", type=str, default="dataset/amazon_instruments_user_text.pkl")

    data_bert = data.add_argument_group("BERT")
    data_bert.add_argument("--bert", action="store_true", help="Whether to use BERT embedding")
    data_bert.add_argument(
        "--bert_embedding_path", type=str, default="dataset/node_embeddings/bert.pt", help="Path to the BERT embedding file"
    )
    data_bert.add_argument(
        "--bert_reduced_dim", type=int, default=64, help="Dimensionality of the reduced BERT embedding"
    )

    module = parser.add_argument_group("Module")
    module.add_argument("--module_type", type=str, default="GCN", help="Type of the module to use")
    module.add_argument("--module_weight_decay", type=float, default=5e-4, help="Weight decay of the module")
    module.add_argument("--module_lr", type=float, default=0.01, help="Learning rate of the module")

    module_gcn = module.add_argument_group("GCN")
    module_gcn.add_argument("--gcn_hidden_dim", type=int, default=16, help="Hidden dimension of the GCN module")
    module_gcn.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout rate of the GCN module")
    module_gcn.add_argument("--gcn_num_layers", type=int, default=2, help="Number of layers of the GCN module")
    module_gcn.add_argument(
        "--gcn_pos_weight", type=float, default=10.0, help="Positive class weight of the GCN module"
    )
    module_gcn.add_argument(
        "--gcn_autoencoder", action="store_true", help="Whether to use autoencoder for BERT embedding"
    )
    
    module_graphsage = module.add_argument_group("graphsage")
    module_graphsage.add_argument("--graphsage_hidden_dim", type=int, default=16, help="Hidden dimension of the GraphSAGE module")
    module_graphsage.add_argument("--graphsage_dropout", type=float, default=0.5, help="Dropout rate of the GraphSAGE module")
    module_graphsage.add_argument("--graphsage_num_layers", type=int, default=2, help="Number of layers of the GraphSAGE module")
    module_graphsage.add_argument(
        "--graphsage_pos_weight", type=float, default=10.0, help="Positive class weight of the GraphSAGE module"
    )

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--trainer_max_epochs", type=int, default=2000, help="Maximum number of epochs to train")
    trainer.add_argument("--seed", type=int, default=42, help="Seed for the trainer")
    trainer.add_argument("--run_name", type=str, default=None, help="Name of the logger")

    args = parser.parse_args()

    _validate_args(args)

    return args
