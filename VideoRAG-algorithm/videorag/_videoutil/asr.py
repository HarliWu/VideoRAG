import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def _load_whisper_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return WhisperModel(model_path, device=device)
    except Exception as e:
        if device == "cuda":
            logging.warning(
                "Failed to initialize faster-whisper on CUDA (%s). Falling back to CPU. "
                "This can happen when CTranslate2 CUDA wheels do not match the local CUDA runtime.",
                e,
            )
            return WhisperModel(model_path, device="cpu")
        raise

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = _load_whisper_model("./faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts
