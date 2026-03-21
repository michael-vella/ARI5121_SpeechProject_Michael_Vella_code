import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

wav_file_one = "datasets/abi-1-corpus/accents/brm_001/female/alw001/shortpassagea_CT.wav"
wav_file_two = "datasets/abi-1-corpus/accents/brm_001/female/alw001/shortpassageb_CT.wav"
wav_file_three = "datasets/abi-1-corpus/accents/shl_001/male/jxw001/shortpassagea_CT.wav"  # Different speaker

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv")
model.eval()

def get_speaker_embedding(wav_path):
    waveform, sample_rate = sf.read(wav_path)
    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state       # (1, T, 768)
    embedding = hidden_states.mean(dim=1)           # (1, 768)
    return embedding.squeeze()                      # (768,)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1-D embedding tensors.
    Returns a float in [-1, 1]. Higher = more similar.
    """
    emb1 = emb1.unsqueeze(0)   # (1, 768) — F.cosine_similarity expects 2D
    emb2 = emb2.unsqueeze(0)   # (1, 768)
    score = F.cosine_similarity(emb1, emb2, dim=1)
    return score.item()


def compare_speakers(path1: str, path2: str, threshold: float = 0.75) -> dict:
    """
    Load two wav files, extract embeddings, compute similarity,
    and make a same/different speaker decision based on a threshold.
    """
    emb1 = get_speaker_embedding(path1)
    emb2 = get_speaker_embedding(path2)
    score = cosine_similarity(emb1, emb2)
    same_speaker = score >= threshold

    return {
        "file_1": path1,
        "file_2": path2,
        "similarity": round(score, 4),
        "threshold": threshold,
        "same_speaker": same_speaker,
    }


def print_result(result: dict):
    verdict = "✅ Same speaker" if result["same_speaker"] else "❌ Different speaker"
    print(f"\n{verdict}")
    print(f"  File 1     : {result['file_1']}")
    print(f"  File 2     : {result['file_2']}")
    print(f"  Similarity : {result['similarity']:.4f}  (threshold: {result['threshold']})")


# --- Same speaker, different passages ---
result_same = compare_speakers(wav_file_one, wav_file_two, threshold=0.75)
print_result(result_same)

# --- Different speakers ---
result_diff = compare_speakers(wav_file_one, wav_file_three, threshold=0.75)
print_result(result_diff)