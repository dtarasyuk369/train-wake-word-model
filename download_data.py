import os
import argparse
import logging

import numpy as np
import scipy.io.wavfile
import scipy.signal
import datasets
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TARGET_SR = 16000


def _extract_audio(row, target_sr=TARGET_SR):
    audio_obj = row["audio"]

    # New API (datasets + torchcodec): AudioDecoder object
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        arr = samples.data.numpy().flatten()
        sr = samples.sample_rate
    # Old API (datasets + soundfile): dict with path/array/sampling_rate
    else:
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = audio_obj["sampling_rate"]

    if sr != target_sr:
        n_samples = int(len(arr) * target_sr / sr)
        arr = scipy.signal.resample(arr, n_samples).astype(np.float32)

    return (arr * 32767).astype(np.int16)


def download_mit_rirs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    if os.listdir(output_dir):
        log.info("MIT RIRs уже скачаны, пропуск")
        return

    log.info("Загрузка MIT RIRs...")
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )

    for i, row in enumerate(tqdm(rir_dataset, desc="MIT RIRs")):
        audio = _extract_audio(row)
        name = f"rir_{i:04d}.wav"
        scipy.io.wavfile.write(os.path.join(output_dir, name), TARGET_SR, audio)


def download_audioset_16k(output_dir: str, max_clips: int = 5000):
    os.makedirs(output_dir, exist_ok=True)

    if os.listdir(output_dir):
        log.info("AudioSet 16kHz уже скачан, пропуск")
        return

    log.info("Загрузка AudioSet balanced (до %d клипов, 16kHz)...", max_clips)
    ds = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True
    )

    for i, row in enumerate(tqdm(ds, desc="AudioSet 16kHz", total=max_clips)):
        if i >= max_clips:
            break
        audio = _extract_audio(row)
        name = f"audioset_{i:06d}.wav"
        scipy.io.wavfile.write(os.path.join(output_dir, name), TARGET_SR, audio)


def download_fma(output_dir: str, n_hours: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    if os.listdir(output_dir):
        log.info("FMA уже скачан, пропуск")
        return

    log.info("Загрузка FMA small (~7 ГБ, ≈%d ч будет сохранено)...", n_hours)
    fma_dataset = datasets.load_dataset(
        "rudraml/fma", name="small", split="train",
        trust_remote_code=True,
    )
    fma_dataset = fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=TARGET_SR))

    target_clips = n_hours * 3600 // 30
    for i in tqdm(range(min(target_clips, len(fma_dataset))), desc="FMA"):
        row = fma_dataset[i]
        audio = _extract_audio(row)
        name = f"fma_{i:06d}.wav"
        scipy.io.wavfile.write(os.path.join(output_dir, name), TARGET_SR, audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--fma-hours",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--audioset-clips",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--skip-rirs",
        action="store_true",
    )
    parser.add_argument(
        "--skip-audioset",
        action="store_true",
    )
    parser.add_argument(
        "--include-fma",
        action="store_true",
        help="Скачать FMA (~7 ГБ фоновой музыки). По умолчанию выключено.",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if not args.skip_rirs:
        download_mit_rirs(os.path.join(args.data_dir, "mit_rirs"))

    if not args.skip_audioset:
        download_audioset_16k(
            output_dir=os.path.join(args.data_dir, "audioset_16k"),
            max_clips=args.audioset_clips,
        )

    if args.include_fma:
        download_fma(
            output_dir=os.path.join(args.data_dir, "fma"),
            n_hours=args.fma_hours,
        )
    else:
        log.info("FMA пропущен (используйте --include-fma для загрузки ~7 ГБ музыки)")

    log.info("Готово!")


if __name__ == "__main__":
    main()
