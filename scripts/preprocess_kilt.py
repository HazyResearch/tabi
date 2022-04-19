"""Convert KILT-formatted jsonlines files to TABi-formatted jsonlines files."""

import argparse
import glob
import logging
import os

import jsonlines
from tqdm import tqdm

from tabi.utils.data_utils import load_entity_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input KILT file to convert. Only needed if input_dir and ouput_dir are NOT provided",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file. Only needed if input_dir and output_dir are NOT provided",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory to read KILT files. This reads all jsonlines files in the directory!",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to write TABi files"
    )
    parser.add_argument(
        "--entity_file", type=str, required=True, help="KILT-E knowledge base path"
    )
    return parser.parse_args()


def convert_kilt_to_tabi(kilt_file, tabi_file, wikiid2eid):
    with jsonlines.open(kilt_file) as f, jsonlines.open(tabi_file, "w") as f_out:
        for line in f:
            # get mentions (if any)
            has_mention_boundaries = ("[START_ENT]" in line["input"]) and (
                "[END_ENT]" in line["input"]
            )
            if has_mention_boundaries:
                meta = line["meta"]
                start_idx = len(meta["left_context"]) + 1
                end_idx = start_idx + len(meta["mention"])
                text = (
                    line["input"].replace("[START_ENT] ", "").replace(" [END_ENT]", "")
                )
                mentions = [[start_idx, end_idx]]
            else:
                text = line["input"]
                mentions = []

            # no labels provided (e.g. test dataset)
            if "output" not in line or not any(
                ["provenance" in o for o in line["output"]]
            ):
                f_out.write(
                    {
                        "text": text,
                        "label_id": [-1],
                        "alt_label_id": [[]],
                        "id": line["id"],
                        "mentions": mentions,
                    }
                )
                continue

            # convert labels from wikipedia page ids to KILT-E entity ids
            all_eids = []
            for o in line["output"]:
                # take first wikipedia id to be the label
                if "provenance" in o:
                    for pair in o["provenance"]:
                        wikiid = pair["wikipedia_id"]
                        # some wikipedia page ids won't have eids if they are in KILT but not in KILT-E (e.g. list pages)
                        eid = int(wikiid2eid.get(wikiid, -1))
                        all_eids.append(eid)

            # get unique entity ids
            all_eids = list(set(all_eids))
            assert len(all_eids) > 0
            f_out.write(
                {
                    "text": text,
                    "label_id": [all_eids[0]],
                    "alt_label_id": [all_eids[1:]],
                    "id": line["id"],
                    "mentions": mentions,
                }
            )


def main(args):
    assert (args.input_file and args.output_file) or (
        args.input_dir and args.output_dir
    ), "Must provide either input_file and output_file OR input_dir and output_dir"

    logger.info("Loading entity data...")
    entity_cache = load_entity_data(args.entity_file)

    # mapping from Wikipedia page ids to KILT-E knowledge base ids
    wikiid2eid = {}
    for eid in entity_cache:
        wikiid2eid[entity_cache[eid]["wikipedia_page_id"]] = eid

    if args.output_dir is not None:
        # make output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # iterate over each file in the input dir
        assert (
            args.input_dir is not None
        ), "Must provide input_dir if output_dir is provided"

        # assumes all jsonlines files in the directory are in KILT format!
        kilt_files = glob.glob(f"{args.input_dir}/*")
        logger.info(f"Found {len(kilt_files)} KILT files.")
        for kilt_file in tqdm(kilt_files, desc="Converting"):
            tabi_file = os.path.join(args.output_dir, os.path.basename(kilt_file))
            convert_kilt_to_tabi(
                kilt_file=kilt_file, tabi_file=tabi_file, wikiid2eid=wikiid2eid
            )

    else:
        logger.info(f"Converting {args.input_file}...")
        convert_kilt_to_tabi(
            kilt_file=args.input_file, tabi_file=args.output_file, wikiid2eid=wikiid2eid
        )
        logger.info(f"Wrote {args.output_file}!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
