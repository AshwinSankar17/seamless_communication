import re
import os
import glob
import json
import soundfile as sf
import tqdm
from joblib import Parallel, delayed
import argparse
from utils_dataset_clean import lang_codes, clean_sentence, DICTS
from utils_dataset_custom_transforms import custom_word_transforms, custom_punct_transforms

SANSKRIT_FIX = ['2251799813688787.wav','2251799813703859.wav','2251799813692126.wav','2251799813686641.wav','2251799813689309.wav','2251799813687387.wav','2251799813688849.wav','2251799813695281.wav','2251799813700347.wav','2251799813693904.wav','2251799813694067.wav','2251799813692501.wav','2251799813687140.wav','2251799813687139.wav','2251799813701729.wav','2251799813703394.wav','2251799813704618.wav','2251799813701291.wav','2251799813713054.wav','2251799813721409.wav','2251799813703924.wav','2251799813711477.wav','2251799813711166.wav','2251799813719392.wav','2251799813705621.wav','2251799813705100.wav','2251799813742679.wav','2251799813775563.wav','2251799813715967.wav','2251799813722635.wav','2251799813714326.wav','2251799813778244.wav','2251799813728160.wav','2251799813754112.wav','2251799813754751.wav','2251799813748855.wav','2251799813748925.wav','2251799813738717.wav','2251799813726449.wav','2251799813749284.wav','2251799813767792.wav','2251799813778379.wav','2251799813740695.wav','2251799813779121.wav','2251799813782048.wav','2251799813764160.wav','2251799813765836.wav','2251799813754272.wav','2251799813764044.wav','2251799813753052.wav','2251799813766111.wav','2251799813763171.wav','2251799813769405.wav','2251799813769535.wav','2251799813797565.wav','2251799813797849.wav','2251799813777793.wav','2251799813785588.wav','2251799813799195.wav','2251799813801380.wav','2251799813778125.wav','2251799813809190.wav','2251799813781817.wav','2251799813808453.wav','2251799813780039.wav','2251799813780551.wav','2251799813781070.wav','2251799813808513.wav','2251799813808704.wav','2251799813792616.wav','2251799813809482.wav','2251799813782159.wav','2251799813783153.wav','2251799813811254.wav','2251799813812522.wav','2251799813793750.wav','2251799813794518.wav']

language_map = {
    "as": "asm",  # Assamese
    "bn": "ben",  # Bengali
    "gu": "guj",  # Gujarati
    "hi": "hin",  # Hindi
    "kn": "kan",  # Kannada
    "ks": "kas",  # Kashmiri
    "ml": "mal",  # Malayalam
    "mr": "mar",  # Marathi
    "ne": "nep",  # Nepali
    "or": "ori",  # Odia
    "pa": "pan",  # Punjabi
    "sa": "san",  # Sanskrit
    "sd": "snd",  # Sindhi
    "ta": "tam",  # Tamil
    "te": "tel",  # Telugu
    "ur": "urd",  # Urdu
    "kok": "kok", # Konkani
    "doi": "doi", # Dogri (3-letter code only; no 2-letter code exists)
    "mai": "mai", # Maithili (3-letter code only; no 2-letter code exists)
    "sat": "sat", # Santali (3-letter code only; no 2-letter code exists)
    "mni": "mni", # Manipuri/Meitei (3-letter code only; no 2-letter code exists)
    "brx": "brx", # Bodo
}


def clean_text(text):
    if text is not None:
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_json(json_path, input_dir, output_dir):
    language = json_path.split('/')[-4]
    wav_path = json_path.replace('.json', '.wav')
    with open(json_path) as reader:
        json_obj = json.load(reader)

    output_root, name = os.path.split(wav_path.replace(input_dir, output_dir))
    if name in SANSKRIT_FIX:
        json_obj['task_name'] = 'Digital Payment Commands'

    if json_obj['task_name'] == 'DOI - Fish Farming':
        json_obj['task_name'] = 'DOI - Animal Husbandry'

    output_wavs = f"{output_root}/audios"
    os.makedirs(output_wavs, exist_ok=True)
    output_wav_path = f'{output_wavs}/{name}'

    transcript_path = f"{output_root}/transcripts"
    extras_path = f"{output_root}/errors"
    os.makedirs(transcript_path, exist_ok=True)
    os.makedirs(extras_path, exist_ok=True)

    wav, sr = sf.read(wav_path)
    assert sr == 16000, 'Audio files must be sampled at 16kHz'
    file_jsons = []
    errors = []

    for e, (verbatim, normalized) in enumerate(zip(json_obj['verbatim'], json_obj['normalized'])):
        lang_id = lang_codes[language.lower()]
        start, end = int(max(verbatim['start'], 0) * sr), int(verbatim['end'] * sr)
        wav_patch = wav[start:end]

        vvalid, vtext, vextras = clean_sentence(
            verbatim['text'], DICTS[lang_id], 
            custom_word_transform=custom_word_transforms.get(lang_id, {}), 
            custom_punct_transform=custom_punct_transforms.get(lang_id, {}), 
            extras=True
        )
        nvalid, ntext, nextras = clean_sentence(
            normalized['text'], DICTS[lang_id], 
            custom_word_transform=custom_word_transforms.get(lang_id, {}), 
            custom_punct_transform=custom_punct_transforms.get(lang_id, {}), 
            extras=True
        )
        
        chunk_path = f'{output_wav_path.replace(".wav", f"_chunk_{e+1}.wav")}'
        err_path = chunk_path.replace('/audios/', '/errors/').replace('.wav', '.err')

        if not (vvalid and nvalid):
            errors.append((err_path, vextras[0] | nextras[0], vtext, verbatim['text'], ntext, normalized['text']))
            continue

        if end - start == 0 or verbatim['speaker_id'] == 1 or len(vtext) == 0 or len(ntext) == 0 or end - start > 30 * sr or end - start < 0.05 * sr:
            continue

        sf.write(chunk_path, wav_patch, sr)
        file_jsons.append(json.dumps({
            "source": {
                "id": os.path.basename(chunk_path).replace(".wav", ""),
                "text": clean_text(verbatim["text"]),
                "lang": language_map[lang_id],
                "audio_local_path": chunk_path,
                "sampling_rate": sr
            },
            "target": {
                "id": os.path.basename(chunk_path).replace(".wav", ""),
                "text": clean_text(normalized["text"]),
                "lang": language_map[lang_id]
            }
        }, ensure_ascii=False))

    if file_jsons:
        with open(f"{transcript_path}/{name.replace('.wav', '.json')}", 'w') as writer:
            writer.write('\n'.join(file_jsons))
            writer.write('\n')

    for fp, a, b, c, d, e in errors:
        with open(fp, 'w') as writer:
            print(a, b, c, d, e, sep='\n', file=writer)


def main():
    parser = argparse.ArgumentParser(description="Process Indic speech datasets.")
    parser.add_argument("--input_folder", required=True, help="Path to input folder containing data.")
    parser.add_argument("--output_folder", required=True, help="Path to output folder for processed data.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel jobs.")
    args = parser.parse_args()

    input_dir = args.input_folder
    output_dir = args.output_folder
    num_workers = args.num_workers

    splits = {
        "train": glob.glob(f"{input_dir}/**/train/**/*.json", recursive=True),
        "val": glob.glob(f"{input_dir}/**/valid/**/*.json", recursive=True),
        "test": glob.glob(f"{input_dir}/**/test/**/*.json", recursive=True)
    }

    for split, json_files in splits.items():
        split_manifest = []
        print(f"Processing {split} data with {len(json_files)} files...")
        split_manifest += Parallel(n_jobs=num_workers, backend='multiprocessing')(
            delayed(parse_json)(j, input_dir, output_dir) for j in tqdm.tqdm(json_files)
        )
        # Flatten the list of lists and save the manifest
        manifest_path = os.path.join(output_dir, f"{split}_manifest.json")
        with open(manifest_path, 'w') as writer:
            for entry in split_manifest:
                writer.write(json.dumps(entry, ensure_ascii=False) + '\n')





if __name__ == '__main__':
    main()