import argparse
import logging
from collections import defaultdict

from string import punctuation
import torch
from termcolor import colored
from transformers import AutoTokenizer
from transformers import logging as hf_logging

from tabi.constants import ENT_START
from tabi.models.biencoder import Biencoder
from tabi.utils.data_utils import load_entity_data
from tabi.utils.utils import load_model, move_dict

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--entity_emb_path", type=str)
parser.add_argument("--entity_file", type=str)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")
args = parser.parse_args()


def preprocess_query(tokenizer, text):
    # Take the input data and make it inference ready
    tokens = tokenizer(
        text,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",  # return as pytorch tensors
        truncation=True,
        max_length=32,
        return_length=True,
        return_offsets_mapping=True,
    )
    context_tokens = defaultdict(list)
    for key in tokens.keys():
        context_tokens[key].append(tokens[key][0])
    context_tokens = {k: torch.stack(v) for k, v in context_tokens.items()}
    return context_tokens


def preprocess_entity(tokenizer, title, description):
    ent_str = title + " " + ENT_START + " " + description
    entity_tokens = tokenizer(
        ent_str,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    return entity_tokens


# load model
logger.info("Loading model...")
model = Biencoder(
    tied=True,
    entity_emb_path=args.entity_emb_path,
    top_k=args.top_k,
    model_name="bert-base-uncased",
    normalize=True,
    temperature=0.05,
)
load_model(model_checkpoint=args.model_checkpoint, device=args.device, model=model)
model.to(args.device)
model.eval()
logger.info("Finished loading model!")

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"additional_special_tokens": [ENT_START]})

if args.entity_emb_path is not None:
    # load entity cache
    logger.info("Loading entity data...")
    entity_cache = load_entity_data(args.entity_file)
    logger.info("Finished loading entity data!")


def pretty_print(ent_data, prob, score):
    print(colored(f"\ntitle: {ent_data['title']}", "grey", "on_cyan"))
    print(f"prob: {round(prob, 5)}")
    print(f"score: {round(score, 5)}")
    print(f"text:{' '.join(ent_data['description'].split(' ')[:150])}")


if args.entity_emb_path is None:
    logger.info(
        "Using entity-input mode. No entity index was provided. To enter a new query, type 'Exit' for entity title. Returns raw score."
    )

while True:
    # ask for input
    text = input(colored("\nInsert query: ", "grey", "on_green"))
    if text.lower() == "exit" or text == "exit()":
        break

    # remove punctuation from end of text
    text = text.rstrip(punctuation)

    query_tokens = preprocess_query(tokenizer=tokenizer, text=text)

    # use index if provided
    if args.entity_emb_path is not None:
        # retrieve candidates
        with torch.no_grad():
            res = model.predict(
                context_data=move_dict(query_tokens, args.device),
                data_id=torch.tensor([-1]),
            )
            assert len(res["probs"]) == 1
            res["probs"] = res["probs"][0].tolist()
            res["indices"] = res["indices"][0].tolist()
            res["scores"] = res["scores"][0].tolist()
            del res["data_id"]

        # return response to user
        for eid, prob, score in zip(res["indices"], res["probs"], res["scores"]):
            pretty_print(entity_cache[eid], prob, score)

    # otherwise, return query for entity info and return raw score
    else:
        exit_code = False
        while not exit_code:
            entity_title = input("\nInsert entity title: ")
            if entity_title.lower() == "exit" or entity_title == "exit()":
                exit_code = True
                break
            entity_description = input("Insert entity description: ")
            entity_tokens = preprocess_entity(
                tokenizer=tokenizer, title=entity_title, description=entity_description
            )
            # print(tokenizer.decode(entity_tokens['input_ids'][0]))
            # compute embeddings and take dot product
            ent_emb = model._embed_entity(move_dict(entity_tokens, args.device))
            query_emb = model._embed_query(move_dict(query_tokens, args.device))
            score = torch.dot(ent_emb.squeeze(), query_emb.squeeze())
            print(f"Score: {round(score.item(), 5)}")
