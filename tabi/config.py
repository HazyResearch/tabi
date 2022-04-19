import argparse

from tabi.utils.utils import str2bool

parser = argparse.ArgumentParser(add_help=False)

general_args = parser.add_argument_group("general_args")
general_args.add_argument(
    "--verbose", type=str2bool, default="False", help="Print debug information"
)
general_args.add_argument(
    "--distributed",
    type=str2bool,
    default="False",
    help="Use distributed data parallel",
)
general_args.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="Local rank. Provided by pytorch torch.distributed.launch script.",
)
general_args.add_argument(
    "--log_dir", type=str, required=True, help="Directory to save log and outputs"
)
general_args.add_argument(
    "--num_workers", type=int, default=4, help="Number of dataloader workers"
)
general_args.add_argument(
    "--gpu", type=int, default=0, help="Device to use (-1 if CPU)"
)
general_args.add_argument("--batch_size", type=int, default=32, help="Batch size")
general_args.add_argument("--seed", type=int, default=1234, help="Seed for training")
general_args.add_argument("--type_file", type=str, help="List of types")

model_args = parser.add_argument_group("model_args")
model_args.add_argument(
    "--tied",
    type=str2bool,
    default="True",
    help="Tie mention and entity encoder weights",
)
model_args.add_argument("--temperature", type=float, default=0.1, help="Temperature")
model_args.add_argument(
    "--model_name",
    type=str,
    default="bert-base-uncased",
    help="Transformer model for initialization",
)
model_args.add_argument(
    "--tokenizer_name",
    type=str,
    default="bert-base-uncased",
    help="Transformer tokenizer",
)
model_args.add_argument(
    "--normalize",
    type=str2bool,
    default=True,
    help="Use L2 normalization for entity and mention embs. If using normalization, a lower temperature value (i.e. 0.1) is recommended.",
)
model_args.add_argument(
    "--add_entity_type_in_description",
    type=str2bool,
    default="False",
    help="Add the entity type in the entity encoding",
)
model_args.add_argument(
    "--max_entity_length",
    type=int,
    default=128,
    help="Max numbers of tokens for entity",
)
model_args.add_argument(
    "--max_context_length",
    type=int,
    default=32,
    help="Max number of tokens for mention context",
)
