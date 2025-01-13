import json
import torch
import torchaudio
from glob import glob
from pathlib import Path
from datasets import Dataset

from fairseq2.data.audio import WaveformToFbankConverter

from seamless_communication.datasets.datatypes import LangPairSample
from seamless_communication.models.unity import load_unity_text_tokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process manifest.json file for dataset preparation.")
    parser.add_argument("manifest_path", type=str, help="Path to the manifest.json file")
    return parser.parse_args()

args = parse_args()
manifest_path = args.manifest_path


ds_src_path = "/data-3/ashwin2/BhasaAnuvaad"

src_allowed_keys = ['id', 'text', 'lang', 'audio_local_path', 'sampling_rate']
tgt_allowed_keys = ['id', 'text', 'lang']

text_tokenizer = load_unity_text_tokenizer("seamlessM4T_v2_large")
text_encoders_per_lang = {}

# train_manifest_paths = glob(f"{ds_src_path}/**/train_manifest.json", recursive=True)
# train_manifest_paths = ["/data-3/ashwin2/BhasaAnuvaad/UGCE-Resources/en2indic/english/train_manifest.json"]
# test_manifest_paths = glob(f"{ds_src_path}/**/test_manifest.json", recursive=True)

# print(train_manifest_paths[0])
# exit()

_fbank_extract_params = {
    "num_mel_bins": 80,
    "waveform_scale": 32768,
    "channel_last": True,
    "standardize": True,
    "device": torch.device("cpu"),
    "dtype": torch.float32,
}
feature_extractor = WaveformToFbankConverter(**_fbank_extract_params)

# print(train_manifest_paths[0])
# exit()

def _get_tokenized_target(text_tokenizer, text_encoders_per_lang, json_sample):
    """Expected sequence is [<eos>, <lang_tok> , ..text tokens.., <eos>]"""
    sample = LangPairSample.from_json(json_sample)
    target_lang = sample.target.lang
    if target_lang not in text_encoders_per_lang:
        text_encoders_per_lang[target_lang] = (
            text_tokenizer.create_encoder(lang=target_lang, mode="target")
        )
    tokens = text_encoders_per_lang[target_lang](sample.target.text)
    eos_idx = text_tokenizer.vocab_info.eos_idx
    tokens = torch.concat([tokens, torch.LongTensor([eos_idx])])
    json_sample['target_tokens'] = tokens
    return json_sample

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

def _get_source_fbank(sample, feature_extractor):
    # print(sample)
    SAMPLE_RATE = 16_000
    wav, sample_rate = torchaudio.load(sample['source']['audio_local_path'])
    assert (
        int(sample_rate) == SAMPLE_RATE
    ), f"sample != {SAMPLE_RATE}, please resample"
    assert len(wav.shape) in (1, 2)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(-1)
    elif wav.shape[0] <= 2:  # channel is first, should be second
        wav = wav.transpose(0, 1)
    sample["fbank"] = feature_extractor(  # type: ignore
        {
            "waveform": wav.to(_fbank_extract_params["device"]),
            "sample_rate": SAMPLE_RATE,
        }
    )["fbank"]
    # sample["fbank"] = WaveformToFbankConverter(**_fbank_extract_params)(  # type: ignore
    #     {
    #         "waveform": wav.to(_fbank_extract_params["device"]),
    #         "sample_rate": SAMPLE_RATE,
    #     }
    # )["fbank"]
    return sample

def collect_fbank_and_tokens(sample, text_tokenizer, text_encoders_per_lang, feature_extractor):
    try:
        sample = _get_source_fbank(sample, feature_extractor)
        sample = _get_tokenized_target(text_tokenizer, text_encoders_per_lang, sample)
    except Exception as e:
        print(f"Failed to load sample path: {sample['source']['audio_local_path']}; {e}")
    finally:
        return sample


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
        return False

def filter_fbanks(sample):
    return not sample.isnan().any().item()

def collated_filter(sample, max_audio_length_sec=15.0, min_audio_length=0.3):
    try:
        wav, sample_rate = torchaudio.load(sample['source']['audio_local_path'])
        length_s: float = max(wav.shape) / sample_rate
        tokens = sample['target_tokens']
        return not (length_s < min_audio_length or length_s > max_audio_length_sec or tokens.shape[-1] >= 4096 or sample['fbank'].isnan().any().item())
    except Exception as e:
        print(f"Failed to load sample path: {sample.source.audio_local_path}; {e}")
        return False

def simple_text_len_filter(sample):
    return len(sample['target']['text']) > 7


print(manifest_path)
alljsonlines = []
with open(manifest_path) as fp_in:
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
ds = Dataset.from_list(alljsonlines).filter(simple_text_len_filter, num_proc=16)
ds = ds.map(lambda sample: collect_fbank_and_tokens(sample, text_tokenizer, text_encoders_per_lang, feature_extractor), num_proc=16)
# ds = ds.map(lambda x: _get_tokenized_target(text_tokenizer, text_encoders_per_lang, x), num_proc=64)
ds.set_format('torch', columns=['fbank', 'target_tokens'], output_all_columns=True)
ds = ds.filter(collated_filter, num_proc=16)
ds.save_to_disk(Path(manifest_path).parent / "precompiled/train", max_shard_size="1GB")
