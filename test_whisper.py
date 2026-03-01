import numpy as np
from faster_whisper import WhisperModel
import faster_whisper
import tokenizers

def test():
    model = WhisperModel("medium", device="cpu", compute_type="auto", download_root="models/stt/")
    print(f"model.model.is_multilingual: {model.model.is_multilingual}")
    
    tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, "transcribe", "en")
    print("sot_sequence:", tokenizer.sot_sequence)

if __name__ == "__main__":
    test()
