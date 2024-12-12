import random
import typing
import argparse
import re
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import torchaudio as ta

import torch
import dataclasses

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
    "Mann-ki-Baat", "NPTEL", "UGCE-Resources", 
    "Vanipedia", "Spoken-Tutorial", "fleurs", "all", "none"
]

ALIGNMENT_THRESHOLD = float(os.environ.get("ALIGNMENT_THRESHOLD", 0.8))
MINING_THRESHOLD = float(os.environ.get("MINING_THRESHOLD", 0.6))

UNITY_TO_FLEURS_LANG_MAPPING = {
    "eng": "en_us",
    "asm": "as_in",
    "ben": "bn_in",
    "guj": "gu_in",
    "hin": "hi_in",
    "kan": "kn_in",
    "mal": "ml_in",
    "mar": "mr_in",
    "ory": "or_in",
    "pan": "pa_in",
    "snd": "sd_in",
    "tam": "ta_in",
    "tel": "te_in",
}

class UnitSpeechTokenizer(SpeechTokenizer):
    MODEL_NAME = "xlsr2_1b_v2"
    KMEANS_MODEL_URI = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    OUTPUT_LAYER_IDX = 34

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.unit_extractor = UnitExtractor(
            model_name_or_card=self.MODEL_NAME,
            kmeans_uri=self.KMEANS_MODEL_URI,
            device=self.device,
        )

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.unit_extractor.predict(
            wav.to(self.device),
            out_layer_idx=self.OUTPUT_LAYER_IDX,
            sample_rate=sample_rate,
        )

def clean_text(text):
    if text is not None:
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text

def alignment_filter(sample):
    if "alignment_score" not in sample:
        return True
    return sample["alignment_score"] >= ALIGNMENT_THRESHOLD

def mining_filter(sample, column):
    mining_column = f"{column.split('_')[0]}_mining_score"
    if mining_column not in sample:
        return True
    return sample[mining_column] >= MINING_THRESHOLD

def _dispatch_download_fleurs(
    source_lang: str,
    split: str,
    save_directory: str,
    hf_cache_dir: str,
):
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )
    os.makedirs(f"{save_directory}/{source_lang}", exist_ok=True)
    manifest_path: str = os.path.join(save_directory, f"{source_lang}/{split}_manifest.json")
    with open(manifest_path, "a") as fp_out:
        for target_lang in UNITY_TO_FLEURS_LANG_MAPPING.keys():
            tokenizer = UnitSpeechTokenizer(device=device)
            dataset_iterator = Speech2SpeechFleursDatasetBuilder(
                source_lang=UNITY_TO_FLEURS_LANG_MAPPING[source_lang],
                target_lang=UNITY_TO_FLEURS_LANG_MAPPING[target_lang],
                dataset_cache_dir=hf_cache_dir,
                speech_tokenizer=tokenizer,
                skip_source_audio=True,  # don't extract units from source audio
                skip_target_audio=True,
                split=split,
            )
            for idx, sample in enumerate(dataset_iterator.__iter__(), start=1):
                # correction as FleursDatasetBuilder return fleurs lang codes
                sample.source.lang = source_lang
                sample.target.lang = target_lang
                sample.target.waveform = None  # already extracted units
                fp_out.write(json.dumps(dataclasses.asdict(sample)) + "\n")
    logger.info(f"Saved {idx} samples for split={split} to {manifest_path}")
    logger.info(f"Manifest saved to: {manifest_path}")

def download_fleurs_en2indic(huggingface_token: str, save_directory: str, hf_cache_dir: str = "~/.cache/huggingface/datasets",):
    for split in ['train']: #, 'validation', 'test']:
        # for lang in tqdm(UNITY_TO_FLEURS_LANG_MAPPING.keys()):
        _dispatch_download_fleurs("eng", split, save_directory, hf_cache_dir=hf_cache_dir)

def download_fleurs_indic2en(huggingface_token: str, save_directory: str, hf_cache_dir: str = "~/.cache/huggingface/datasets",):
    for split in ['train']: #, 'validation', 'test']:
        for lang in tqdm([x for x in UNITY_TO_FLEURS_LANG_MAPPING.keys() if x != "eng"]):
            _dispatch_download_fleurs(lang, split, save_directory, hf_cache_dir=hf_cache_dir)

def _dispatch_prepare_en2indic(
    dataset: str, 
    huggingface_token: str, 
    save_directory: str, 
    hf_cache_dir: str = "~/.cache/huggingface/datasets", 
    col_map: dict = None, 
    filter_fn: typing.Callable = alignment_filter
):
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
    ds = load_dataset(dataset, "en2indic", token=huggingface_token, cache_dir=hf_cache_dir, num_proc=32)
    ds_str = dataset.replace("ai4bharat/", "")
    error_log = open("error.log", "a")
    # Remap column names
    if col_map is not None:
        ds = ds.rename_columns(col_map)
    # Process split
    for split in ds:
        os.makedirs(os.path.join(save_directory, f"{ds_str}/{subset}/english/wavs"), exist_ok=True)
        manifest_path = os.path.join(save_directory, f"{ds_str}/{subset}/english/manifest.json")

        with open(manifest_path, "w") as f:
            logger.info(f"Preparing English split...")
            # ds_iterator = iter(ds[split])
            pbar = tqdm(total=len(ds[split]))
            for idx in range(len(ds[split])):
                try:
                    # sample = next(ds_iterator)
                    sample = ds[split][idx]
                    if filter_fn(sample):
                        if "audio_filepath" in sample:
                            audio_fp = os.path.basename(sample['audio_filepath']).replace(".wav", "")
                            filename = os.path.basename(sample['audio']['path']).replace(".wav", "")
                            filename = f"{audio_fp}_{filename}"
                        else:
                            filename = os.path.basename(sample['audio']['path']).replace(".wav", "")
                        save_filepath = f"{save_directory}/{ds_str}/{subset}/english/wavs/{filename}.wav"
                        if not os.path.exists(save_filepath):
                            audio = librosa.resample(sample['audio']['array'], orig_sr=sample['audio']["sampling_rate"], target_sr=16_000)
                            sf.write(save_filepath, audio, samplerate=16_000)
                        # load to check for decode error
                        audio, sr = ta.load(save_filepath)
                        if "UGCE" in ds_str:
                            a_dur = round(audio.shape[-1] / sr, 1)
                            sample_dur = round(sample['duration'], 1)
                            assert abs(a_dur - sample_dur) < 0.2, f"{a_dur} vs {sample_dur}" 
                        if sample.get("text"):
                            f.write(json.dumps({
                                "source": {
                                    "id": f"segment_{filename}",
                                    "text": sample.get("text"),
                                    "lang":"eng",
                                    "audio_local_path": save_filepath,
                                    "sampling_rate": 16_000,
                                },
                                "target": {
                                    "id": f"segment_{filename}",
                                    "text": sample.get("text"),
                                    "lang": "eng",
                                }
                                }) + "\n")
                        for column, lang_code in columns.items():
                            if sample.get(column) and alignment_filter(sample):
                                tgt_text = clean_text(sample[column])
                                f.write(json.dumps({
                                "source": {
                                    "id": f"segment_{filename}",
                                    "text": clean_text(sample.get("text", None)),
                                    "lang":"eng",
                                    "audio_local_path": save_filepath,
                                    "sampling_rate": 16_000,
                                },
                                "target": {
                                    "id": f"segment_{filename}",
                                    "text": tgt_text,
                                    "lang": lang_code,
                                }
                                }) + "\n")
                    # pbar.update(1)
                except Exception as e:
                    logging.error(f"Skipping index {idx} due to Unhandled error {e}.")
                    error_log.write(f"{save_filepath}\n")
                pbar.update(1)
                    
    pbar.close()
    error_log.close()
    logger.info(f"Manifest for {dataset}-eng2indic saved to: {manifest_path}")

def _dispatch_prepare_indic2en(
        dataset: str, 
        huggingface_token: str, 
        save_directory: str, 
        hf_cache_dir: str = "~/.cache/huggingface/datasets", 
        col_map: dict = None, 
        filter_fn: typing.Callable = alignment_filter
    ):
    subset = "indic2en"
    splits = {
        "assamese": "asm",
        "bengali": "ben",
        "gujarati": "guj",
        "hindi": "hin",
        "kannada": "kan",
        "malayalam": "mal",
        "marathi": "mar",
        "manipuri": "mni",
        "nepali": "npi",
        "odia": "ory",
        "punjabi": "pan",
        "tamil": "tam",
        "telugu": "tel",
        "urdu": "urd",
    }
    ds = load_dataset(dataset, "indic2en", token=huggingface_token, cache_dir=hf_cache_dir, num_proc=32)
    ds_str = dataset.replace("ai4bharat/", "")
    error_log = open("error.log", "a")
    if col_map is not None:
        ds = ds.rename_columns(col_map)
    for split in ds:
        os.makedirs(os.path.join(save_directory, f"{ds_str}/{subset}/{split}/wavs"), exist_ok=True)
        manifest_path = os.path.join(save_directory, f"{ds_str}/{subset}/{split}/manifest.json")
        
        with open(manifest_path, "w") as f:
            logger.info(f"Preparing {split} split...")
            pbar = tqdm(total=len(ds[split]))
            # for idx, sample in tqdm(enumerate(ds[split])):
            for idx in range(len(ds[split])):
                try:
                    # sample = next(ds_iterator)
                    sample = ds[split][idx]
                    if filter_fn(sample) and sample.get('en_text') and mining_filter(sample, "en_text"):
                        if "audio_filepath" in sample:
                            audio_fp = os.path.basename(sample['audio_filepath']).replace(".wav", "")
                            filename = os.path.basename(sample['audio']['path']).replace(".wav", "")
                            filename = f"{audio_fp}_{filename}"
                        else:
                            filename = os.path.basename(sample['audio']['path']).replace(".wav", "")
                        # print(sample['audio']['path'])
                        save_filepath = f"{save_directory}/{ds_str}/{subset}/{split}/wavs/{filename}.wav"
                        if not os.path.exists(save_filepath):
                            audio = librosa.resample(sample['audio']['array'], orig_sr=sample['audio']["sampling_rate"], target_sr=16_000)
                            sf.write(save_filepath, audio, samplerate=16_000)
                        
                        ta.load(save_filepath)
                        if sample.get("text"):
                            f.write(json.dumps({
                                "source": {
                                    "id": f"segment_{filename}",
                                    "text": clean_text(sample.get("text")),
                                    "lang":splits[split],
                                    "audio_local_path": save_filepath,
                                    "sampling_rate": 16_000,
                                },
                                "target": {
                                    "id": f"segment_{filename}",
                                    "text": clean_text(sample.get("text")),
                                    "lang": splits[split],
                                }
                                }) + "\n")

                        f.write(json.dumps({
                        "source": {
                            "id": f"segment_{filename}",
                            "text": clean_text(sample.get("text", None)),
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
                    # pbar.update(1)
                except Exception as e:
                    logging.error(f"Skipping index {idx} due to Unhandled error {e}.")
                    error_log.write(f"{save_filepath}\n")
                pbar.update(1)
    pbar.close()
    error_log.close()
    logger.info(f"Manifest for {dataset}-eng2indic saved to: {manifest_path}")

def download_mkb(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    # os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {"chunked_audio_filepath": "audio"}
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory, hf_cache_dir, col_map)
        _dispatch_prepare_en2indic("ai4bharat/Mann-ki-Baat", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for Mann-ki-Baat dataset")

def download_word_project(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    # os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {"chunked_audio_filepath": "audio"}
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/WordProject", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/WordProject", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/WordProject", huggingface_token, save_directory, hf_cache_dir, col_map)
        _dispatch_prepare_en2indic("ai4bharat/WordProject", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for WordProject dataset")

def download_nptel(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    # os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {"chunked_audio_filepath": "audio"}
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/NPTEL", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/NPTEL", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/NPTEL", huggingface_token, save_directory, hf_cache_dir, col_map)
        _dispatch_prepare_en2indic("ai4bharat/NPTEL", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for NPTEL dataset")

def download_ugce(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    # os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {
        "Gujarati_translation": "gu_text", "Hindi_translation": "hi_text", "Telugu_translation": "te_text",
        "Tamil_translation": "ta_text", "Malayalam_translation": "ml_text", "Kannada_translation": "kn_text",
        "Bangla_translation": "bn_text", "Marathi_translation": "mr_text", "chunked_audio_filepath": "audio"
    }
    if subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/UGCE-Resources", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_en2indic("ai4bharat/UGCE-Resources", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for UGCE-Resources dataset")

def download_vanipedia(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    # os.makedirs(os.path.join(save_directory, f"{subset}/wavs"), exist_ok=True)
    col_map = {
        "gu_translation": "gu_text", "hi_translation": "hi_text", "te_translation": "te_text",
        "ta_translation": "ta_text", "ml_translation": "ml_text", "kn_translation": "kn_text",
        "bn_translation": "bn_text", "mr_translation": "mr_text", "pa_translation": "pa_text",
        "or_translation": "or_text", "as_translation": "as_text", "ur_translation": "ur_text",
        "ne_translation": "ne_text", "chunked_audio_filepath": "audio"
    }
    if subset == "en2indic":
        _dispatch_prepare_en2indic("ai4bharat/Vanipedia", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_en2indic("ai4bharat/Vanipedia", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for Vanipedia dataset")

def download_spoken_tutorial(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    col_map = {
        "mp3_path": "audio", "text": "en_text"
    }
    if subset == "indic2en":
        _dispatch_prepare_indic2en("ai4bharat/Spoken-Tutorial", huggingface_token, save_directory, hf_cache_dir, col_map)
    elif subset == "all":
        _dispatch_prepare_indic2en("ai4bharat/Spoken-Tutorial", huggingface_token, save_directory, hf_cache_dir, col_map)
    else:
        raise ValueError(f"{subset} does not exist for Spoken-Tutorial dataset")

def download_fleurs(subset: str, huggingface_token: str, save_directory: str, hf_cache_dir: str):
    if subset == "en2indic":
        download_fleurs_indic2en(huggingface_token, save_directory, hf_cache_dir)
    if subset == "indic2en":
        download_fleurs_indic2en(huggingface_token, save_directory, hf_cache_dir)
    elif subset == "all":
        download_fleurs_en2indic(huggingface_token, save_directory, hf_cache_dir)
        download_fleurs_indic2en(huggingface_token, save_directory, hf_cache_dir)
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
        choices=SUPPORTED_DATASETS,
        help="HuggingFace name of the dataset to prepare (e.g., 'dataset_name').",
    )
    
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["indic2en", "en2indic", "all"],  # Use 'choices' for validation.
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
        "--hf_cache_dir",
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
        "--do_split",
        action="store_true",
        help="Enable this flag to split each dataset in the save_directory into train and test sets."
    )

    parser.add_argument(
        "--test_duration",
        type=float,
        required=False,
        default=1200.0,
        help="Maximum total duration (in seconds) for the test set. Data points are randomly selected until this threshold is reached."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility of the train-test split. Ensures consistent results across multiple runs."
    )
    
    return parser


def calculate_audio_duration(audio_path: str) -> float:
    """
    Calculate the duration of an audio file using torchaudio.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        float: Duration of the audio file in seconds.
    """
    try:
        info = ta.info(audio_path)
        return info.num_samples / info.sample_rate
    except Exception as e:
        print(f"Error calculating duration for {audio_path}: {e}")
        return 0.0


def train_test_split_by_duration(data: list, test_duration_threshold: float, seed: int = 42) -> typing.Tuple[typing.List, typing.List]:
    """
    Split the dataset into train and test sets by randomly sampling data points for the test set
    until the test set reaches the specified duration threshold. The split is deterministic if a seed is provided.

    Args:
        data (list): List of data samples.
        test_duration_threshold (float): Target total duration for the test set (in seconds).
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_set, test_set)
    """
    random.seed(seed)  # Set the random seed for reproducibility
    remaining_data = data[:]  # Copy the original list to avoid modifying it
    test_set = []
    test_duration = 0.0
    offset = 10

    while test_duration < test_duration_threshold and remaining_data:
        # Randomly select a sample
        sample = random.choice(remaining_data)
        audio_path = sample['source']['audio_local_path']
        duration = calculate_audio_duration(audio_path)

        if (test_duration + duration) <= (test_duration_threshold + offset):
            test_set.append(sample)
            test_duration += duration
            remaining_data.remove(sample)  # Remove the selected sample from the pool
        else:
            break

    train_set = remaining_data  # Remaining samples are the train set
    return train_set, test_set


def split_manifest_files(directory: Path, test_duration_threshold: float, seed: int) -> None:
    """
    Split each `manifest.json` file in the directory into train and test sets based on a duration threshold.
    Save the resulting train and test manifests in the same directory as the original manifest.

    Args:
        directory (Path): The root directory to search for `manifest.json` files.
        test_duration_threshold (float): Target total duration for the test set (in seconds).
    """
    jsonl_files = list(directory.rglob("manifest.json"))

    jsonl_files = list(filter(lambda x: "fleurs" not in x, jsonl_files))

    print(f"Found {len(jsonl_files)} `manifest.json` files in {directory} for splitting.")

    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file}...")
        try:
            # Read and load data from the manifest
            with open(jsonl_file, "r", encoding="utf-8") as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]

            print(f"Total samples in {jsonl_file}: {len(data)}")

            # Split into train and test sets
            train_set, test_set = train_test_split_by_duration(data, test_duration_threshold, seed=seed)

            # Define output file paths
            output_train_file = jsonl_file.parent / "train_manifest.json"
            output_test_file = jsonl_file.parent / "test_manifest.json"

            # Save train manifest
            with open(output_train_file, "w", encoding="utf-8") as f_train:
                for record in train_set:
                    json.dump(record, f_train, ensure_ascii=False)
                    f_train.write("\n")

            # Save test manifest
            with open(output_test_file, "w", encoding="utf-8") as f_test:
                for record in test_set:
                    json.dump(record, f_test, ensure_ascii=False)
                    f_test.write("\n")

            print(f"Train manifest saved to {output_train_file} with {len(train_set)} samples.")
            print(f"Test manifest saved to {output_test_file} with {len(test_set)} samples.")

        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")


def main() -> None:
    args = init_parser().parse_args()
    
    # Validate HuggingFace token if required
    # if args.name in ["speechcolab/gigaspeech", "WordProject"]:
    assert args.huggingface_token is not None, \
        f"A HuggingFace token is required for {args.name}. Please provide it using `--huggingface_token`."

    # Dispatch processing based on the dataset name
    if args.name == "Mann-ki-Baat":
        download_mkb(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "WordProject":
        download_word_project(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "NPTEL":
        download_nptel(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "UGCE-Resources":
        download_ugce(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "Vanipedia":
        download_vanipedia(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "Spoken-Tutorial":
        download_spoken_tutorial(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "fleurs":
        download_fleurs(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    elif args.name == "all":
        download_mkb(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
        download_word_project(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
        download_ugce(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
        download_vanipedia(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
        download_spoken_tutorial(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
        download_fleurs(args.direction, args.huggingface_token, args.save_dir, args.hf_cache_dir)
    else:
        pass
    
    if args.do_split:
        split_manifest_files(args.save_dir, args.test_duration, args.seed)

if __name__ == "__main__":
    main()