import json
import torch
import torchaudio
from glob import glob
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from seamless_communication.datasets.datatypes import LangPairSample

from seamless_communication.models.unity import load_unity_text_tokenizer

ds_src_path = "/home/MLTL-RESEARCH/draj/draj/data/parallel/speechtranslation/seamless_communication/src/seamless_communication/cli/m4t/finetune/data/BhasaAnuvaad"

src_allowed_keys = ['id', 'text', 'lang', 'audio_local_path', 'sampling_rate']
tgt_allowed_keys = ['id', 'text', 'lang']

text_tokenizer = load_unity_text_tokenizer("seamlessM4T_v2_large")
text_encoders_per_lang = {}

train_manifest_paths = glob(f"{ds_src_path}/**/train_manifest.json", recursive=True)
test_manifest_paths = glob(f"{ds_src_path}/**/test_manifest.json", recursive=True)

def _get_tokenized_target_text(text_tokenizer, text_encoders_per_lang, sample):
    """Expected sequence is [<eos>, <lang_tok> , ..text tokens.., <eos>]"""
    target_lang = sample.target.lang
    if target_lang not in text_encoders_per_lang:
        text_encoders_per_lang[target_lang] = (
            text_tokenizer.create_encoder(lang=target_lang, mode="target")
        )
    tokens = text_encoders_per_lang[target_lang](sample.target.text)
    eos_idx = text_tokenizer.vocab_info.eos_idx
    tokens = torch.concat([tokens, torch.LongTensor([eos_idx])])
    return tokens


def _is_long_src_audio_tgt_text(sample, text_tokenizer, text_encoders_per_lang, max_audio_length_sec, min_audio_length=0.3):
    # HACK:: causes errored audios to be excluded but this is difficult to follow
    try:
        sample = LangPairSample.from_json(sample)
        wav, sample_rate = torchaudio.load(sample.source.audio_local_path)
        length_s: float = max(wav.shape) / sample_rate
        tokens = _get_tokenized_target_text(text_tokenizer, text_encoders_per_lang, sample)
        return not (length_s < min_audio_length or length_s > max_audio_length_sec or tokens.shape[-1] >= 4096)
    except Exception as e:
        print(f"Failed to load sample path: {sample.source.audio_local_path}; {e}")
        exit()
        return False

train_ds = []
for ds_path in train_manifest_paths:
    alljsonlines = []
    with open(ds_path) as fp_in:
        for line in fp_in:
            try:
                jsonlline = json.loads(line)
                ## Now we need to make it consistent for jsonlline["source"] and jsonlline["target"]
                jsonlline["source"] = {key: value for key, value in jsonlline["source"].items() if key in src_allowed_keys}
                jsonlline["target"] = {key: value for key, value in jsonlline["target"].items() if key in tgt_allowed_keys}
                ## Make id as string
                jsonlline["source"]["id"] = str(jsonlline["source"]["id"])
                jsonlline["target"]["id"] = str(jsonlline["target"]["id"])
                alljsonlines.append(jsonlline)
            except:
                continue
        ds = Dataset.from_list(alljsonlines).filter(
                lambda x: _is_long_src_audio_tgt_text(
                x, text_tokenizer, text_encoders_per_lang, 15.0
            ), 
            num_proc=64
        )
        ds.save_to_disk(Path(ds_path).parent / "precompiled/train", max_shard_size="1GB")
#         train_ds.append(ds)

# train_ds = concatenate_datasets(train_ds)

# train_ds.save_to_disk(f"{ds_save_path}/train")

test_ds = []
for ds_path in test_manifest_paths:
    alljsonlines = []
    with open(ds_path) as fp_in:
        for line in fp_in:
            jsonlline = json.loads(line)
            ## Now we need to make it consistent for jsonlline["source"] and jsonlline["target"]
            jsonlline["source"] = {key: value for key, value in jsonlline["source"].items() if key in src_allowed_keys}
            jsonlline["target"] = {key: value for key, value in jsonlline["target"].items() if key in tgt_allowed_keys}
            ## Make id as string
            jsonlline["source"]["id"] = str(jsonlline["source"]["id"])
            jsonlline["target"]["id"] = str(jsonlline["target"]["id"])
            alljsonlines.append(jsonlline)
        ds = Dataset.from_list(alljsonlines).filter(
                lambda x: _is_long_src_audio_tgt_text(
                x, text_tokenizer, text_encoders_per_lang, 15.0
            ), 
            num_proc=64
        )
        ds.save_to_disk(Path(ds_path).parent / "precompiled/test", max_shard_size="1GB")
#         test_ds.append(ds)

# test_ds = concatenate_datasets(test_ds)

# test_ds.save_to_disk(f"{ds_save_path}/test")