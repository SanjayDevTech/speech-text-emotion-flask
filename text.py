import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline


classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

def speech_to_text(data, sr):
    input_values = processor(data, return_tensors="pt", padding="longest", sampling_rate=sr).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    text = transcription[0]
    return text

def predict_text_emotion(data, sr):
    text = speech_to_text(data, sr)
    return classifier(text)[0]