import os
import sys

import time
import torch
import torchaudio
import joblib
import librosa


def load_model(path="random_forest_3.pkl"):
    return joblib.load(path)


resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)


def rms_normalize(audio: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(audio * audio))
    return audio / (rms + 1e-8)


def cut(audio: torch.Tensor, sr: int, lower_cut=0.5, upper_cut=9.5):
    start = int(lower_cut * sr)
    end = int(upper_cut * sr)
    return audio[start:end]



mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 40
    }
)

def mfcc_stats_with_deltas(mfcc):
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    features = torch.cat([
        mfcc.mean(dim=1), mfcc.std(dim=1),
        delta.mean(dim=1), delta.std(dim=1),
        delta2.mean(dim=1), delta2.std(dim=1)
    ])

    return features


def load_audio(path):
    audio, sr = librosa.load(path)
    return torch.from_numpy(audio), sr


def process_audio(audio):
    audio = audio.squeeze(0)

    audio = resampler(audio)


    # Preprocessing
    audio = rms_normalize(audio)
    audio = cut(audio, 16000)

    # Feature extraction
    mfcc = mfcc_transform(audio)
    features = mfcc_stats_with_deltas(mfcc)

    return features.numpy()

def get_sorted_files(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".wav")]

    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    return [os.path.join(data_path, f) for f in files]


def run_inference(data_path):
    model = load_model()

    files = get_sorted_files(data_path)

    predictions = []
    times = []

    for file in files:
        audio, sr = load_audio(file)

        start = time.time()

        features = process_audio(audio)
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]

        end = time.time()

        predictions.append(pred)
        times.append(round(end - start, 3))

    return predictions, times


def save_results(predictions, times):
    with open("results.txt", "w") as f:
        for p in predictions:
            f.write(f"{p}\n")

    with open("time.txt", "w") as f:
        for t in times:
            f.write(f"{t}\n")




if __name__ == "__main__":
    data_path = sys.argv[1]

    predictions, times = run_inference(data_path)

    save_results(predictions, times)           