import torchaudio
import torch
import torchaudio.functional as F

def extract_mfcc_torch(audio: torch.Tensor, sr, n_mfcc=13):
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 40
        }
    )

    return mfcc_transform(audio)


def mfcc_stats_torch(mfcc):

    mean = mfcc.mean(dim=1)
    std = mfcc.std(dim=1)

    return torch.cat([mean, std])


def mfcc_stats_with_deltas_torch(mfcc):

    delta = F.compute_deltas(mfcc)
    delta2 = F.compute_deltas(delta)

    return torch.cat([
        mfcc.mean(dim=1), mfcc.std(dim=1),
        delta.mean(dim=1), delta.std(dim=1),
        delta2.mean(dim=1), delta2.std(dim=1)
    ])