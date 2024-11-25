import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

import torch

from datasets import load_dataset
from seamless_communication.datasets.huggingface import (
    Speech2SpeechFleursDatasetBuilder,
    SpeechTokenizer,
)
from seamless_communication.models.unit_extractor import UnitExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")


SUPPORTED_DATASETS = [
    "Mann-ki-Baat", "WordProject", "NPTEL", 
    "UGCE-Resources", "Vanipedia", "Spoken-Tutorial"
]

def rename_columns(dataset, col_map):
    for key, value in col_map.items():
        dataset = dataset.rename_column(key, value)
    return dataset

def _dispatch_prepare_en2indic(dataset: str, huggingface_token: str, save_directory: str, col_map: dict = None):
    subset = "en2indic"
    columns = {
        "as_text": "asm",
        "bn_text": "ben",
        "gu_text": "guj",
        "hi_text": "hin",
        "kn_text": "kan",
        "ml_text": "mal",
        "mni_text": "mni",
        "mr_text": "mar",
        "ne_text": "npi",
        "or_text": "ory",
        "pa_text": "pan",
        "ta_text": "tam",
        "te_text": "tel",
        "ur_text": "urd"
    }
    ds = load_dataset(dataset, "en2indic", token=huggingface_token).rename_column("chunked_audio_filepath", "audio")
    if col_map is not None:
        ds = rename_columns(ds, col_map)
    for split in ds:
        manifest_path = os.path.join(save_directory, f"{subset}/train_manifest.json")
        logger.info(f"Preparing {split} split...")
        with open(manifest_path, "w") as f:
            for sample in tqdm(ds[split]):
                filename = os.path.basename(sample['audio']['path']).split(".")[0]
                save_filepath = f"{save_directory}/{subset}/wavs/{filename}.wav"
                sf.write(save_filepath, sample['audio']["array"], samplerate=sample['audio']["sampling_rate"])
                for column, lang_code in columns.items():
                    if column in sample and sample[column]:
                        f.write(json.dumps({
                        "source": {
                            "id": f"segment_{filename}",
                            "text": sample["text"],
                            "lang":"eng",
                            "audio_local_path": save_filepath,
                            "sampling_rate": sample["audio"]["sampling_rate"],
                        },
                        "target": {
                            "id": f"segment_{filename}",
                            "text": sample[column],
                            "lang": lang_code,
                        }
                        }) + "\n")
        logger.info(f"Manifest for {dataset}-eng2indic saved to: {manifest_path}")

def _dispatch_prepare_indic2en(dataset: str, huggingface_token: str, save_directory: str):
    subset = "en2indic"
    splits = {
        "assamese": "asm",
        "bengali": "ben",
        "gujarati": "guj",
        "hindi": "hin",
        "kannada": "kan",
        "malayalam": "mal",
        "marathi": "mar",
        "manipuri": "mni"
        "nepali": "npi"
        "odia": "ory",
        "punjabi": "pan",
        "tamil": "tam",
        "telugu": "tel",
        "urdu": "urd",
    }
    ds = load_dataset(dataset, "indic2en", token=huggingface_token).rename_column("chunked_audio_filepath", "audio")
    for split in ds:
        manifest_path = os.path.join(save_directory, f"{subset}/train_manifest.json")
        logger.info(f"Preparing {split} split...")
        with open(manifest_path, "w") as f:
            for sample in tqdm(ds[split]):
                filename = os.path.basename(sample['audio']['path']).split(".")[0]
                save_filepath = f"{save_directory}/{subset}/wavs/{filename}.wav"
                sf.write(save_filepath, sample['audio']["array"], samplerate=sample['audio']["sampling_rate"])
                f.write(json.dumps({
                "source": {
                    "id": f"segment_{filename}",
                    "text": sample["text"],
                    "lang": splits[split],
                    "audio_local_path": save_filepath,
                    "sampling_rate": sample["audio"]["sampling_rate"],
                },
                "target": {
                    "id": f"segment_{filename}",
                    "text": sample["en_text"],
                    "lang": "eng",
                }
                }) + "\n")
        logger.info(f"Manifest for {dataset}-eng2indic saved to: {manifest_path}")

def download_mkb(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory)
        _dispatch_prepare_en2indic("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory)
    else:
        raise ValueError(f"{subset} does not exist for Mann-ki-Baat dataset")

def download_word_project(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/WordProject", huggingface_token, save_directory)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/WordProject", huggingface_token, save_directory)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/WordProject", huggingface_token, save_directory)
        _dispatch_prepare_en2indic("ai4bharat/WordProject", huggingface_token, save_directory)
    else:
        raise ValueError(f"{subset} does not exist for WordProject dataset")

def download_nptel(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/NPTEL", huggingface_token, save_directory)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/NPTEL", huggingface_token, save_directory)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/NPTEL", huggingface_token, save_directory)
        _dispatch_prepare_en2indic("ai4bharat/NPTEL", huggingface_token, save_directory)
    else:
        raise ValueError(f"{subset} does not exist for NPTEL dataset")

def download_ugce(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {
        "Gujarati_translation": "gu_text", "Hindi_translation": "hi_text", "Telugu_translation": "te_text",
        "Tamil_translation": "ta_text", "Malayalam_translation": "ml_text", "Kannada_translation": "kn_text",
        "Bangla_translation": "bn_text", "Marathi_translation": "mr_text"
    }
    if subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/UGCE-Resources", huggingface_token, save_directory, col_map)
    elif subset == "all":
        _dispatch_prepare_en2indic("ai4bharat/UGCE-Resources", huggingface_token, save_directory, col_map)
    else:
        raise ValueError(f"{subset} does not exist for UGCE-Resources dataset")

def download_vanipedia(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {
        "gu_translation": "gu_text", "hi_translation": "hi_text", "te_translation": "te_text",
        "ta_translation": "ta_text", "ml_translation": "ml_text", "kn_translation": "kn_text",
        "bn_translation": "bn_text", "mr_translation": "mr_text", "pa_translation": "pa_text",
        "or_translation": "or_text", "as_translation": "as_text", "ur_translation": "ur_text",
        "ne_translation": "ne_text"
    }
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/Vanipedia", huggingface_token, save_directory)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/Vanipedia", huggingface_token, save_directory)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/Vanipedia", huggingface_token, save_directory)
        _dispatch_prepare_en2indic("ai4bharat/Vanipedia", huggingface_token, save_directory)
    else:
        raise ValueError(f"{subset} does not exist for Vanipedia dataset")

def download_spoken_tutorial(subset: str, huggingface_token: str, save_directory: str):
    os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/Spoken-Tutorial", huggingface_token, save_directory)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/Spoken-Tutorial", huggingface_token, save_directory)
    else:
        raise ValueError(f"{subset} does not exist for Spoken-Tutorial dataset")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper script to download the BhasaAnuvaad training dataset, extract units "
            "from target audio, and save the dataset as a manifest compatible with `finetune.py`."
        )
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="HuggingFace name of the dataset to prepare (e.g., 'dataset_name').",
    )
    
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["indic2en", "en2indic"],  # Use 'choices' for validation.
        help=(
            "Translation direction for the dataset preparation:\n"
            "  - 'indic2en': Indic language to English.\n"
            "  - 'en2indic': English to Indic language.\n"
            "  - 'all': English to Indic and Indic to English.\n"
        ),
    )
    
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help=(
            "Directory where the datasets will be stored, including HuggingFace dataset "
            "cache files. Ensure the path exists and is writable."
        ),
    )
    
    parser.add_argument(
        "--huggingface_token",
        type=str,
        required=False,
        default=None,
        help=(
            "Your HuggingFace token for authentication. This is necessary for "
            "downloading restricted datasets such as GigaSpeech."
        ),
    )

    parser.add_argument(
        "--collate",
        action="store_true",
        help="Flag to collate all JSON files in the save directory into a single JSON file.",
    )
    
    return parser

def collate_json_files(directory: Path, output_file: Path) -> None:
    """
    Recursively search for `.jsonl` files in the given directory and concatenate their contents.

    Args:
        directory (Path): The root directory to search for `.jsonl` files.
        output_file (Path): The file where the concatenated results will be saved as a `.jsonl`.
    """
    jsonl_files = list(directory.rglob("*.json"))
    combined_data = []

    print(f"Found {len(jsonl_files)} JSONL files in {directory} for collation.")

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Ignore empty lines
                        try:
                            data = json.loads(line)
                            combined_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line in {jsonl_file}: {e}")
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for record in combined_data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")  # Write each record as a new line in the output file

    print(f"Collated JSONL data saved to {output_file}.")



def main() -> None:
    args = init_parser().parse_args()
    
    # Validate the dataset name
    assert args.name in SUPPORTED_DATASETS, \
        f"The only supported datasets are `{SUPPORTED_DATASETS}`. Please use one of these in `--name`."

    # Validate HuggingFace token if required
    if args.name in ["speechcolab/gigaspeech", "WordProject"]:
        assert args.huggingface_token is not None, \
            f"A HuggingFace token is required for {args.name}. Please provide it using `--huggingface_token`."

    # Dispatch processing based on the dataset name
    if args.name == "Mann-ki-Baat":
        download_mkb(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "WordProject":
        download_word_project(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "NPTEL":
        download_nptel(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "UGCE-Resources":
        download_ugce(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "Vanipedia":
        download_vanipedia(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "Spoken-Tutorial":
        download_spoken_tutorial(args.direction, args.huggingface_token, args.save_dir)
    elif args.name == "all":
        download_mkb(args.direction, args.huggingface_token, args.save_dir)
        download_word_project(args.direction, args.huggingface_token, args.save_dir)
        download_ugce(args.direction, args.huggingface_token, args.save_dir)
        download_vanipedia(args.direction, args.huggingface_token, args.save_dir)
        download_spoken_tutorial(args.direction, args.huggingface_token, args.save_dir)
    else:
        raise ValueError(f"Unhandled dataset: {args.name}")
    
    if args.collate:
        collate_output_file = args.save_dir / "collated_train_manifest.json"
        collate_json_files(Path(args.save_dir), Path(collate_output_file))

if __name__ == "__main__":
    main()