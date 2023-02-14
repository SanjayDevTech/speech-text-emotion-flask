from flask import Flask, request, send_file
from flask_cors import CORS
import os
import base64
import librosa
import pathlib
import noisereduce as nr
from speech import predict_speech_emotion
from text import predict_text_emotion
import soundfile as sf

app = Flask(__name__)
CORS(app)

located_dir = pathlib.Path(__file__).parent.absolute()
temp_dir = os.path.join(located_dir,"temp")

if(not os.path.exists(temp_dir)):
    os.mkdir(temp_dir)

@app.get('/health')
def health():
    return "OK"

@app.get('/temp')
def temp():
    return send_file(os.path.join(temp_dir,"tempaudio.wav"), mimetype="audio/wav", download_name="tempaudio.wav")

@app.get('/reduced')
def reduced():
    return send_file(os.path.join(temp_dir,"reduced.wav"), mimetype="audio/wav", download_name="reduced.wav")

@app.post("/")
def index():
    body = request.json
    data = body["file"]
    # img = body["image"]
    wav_path = os.path.join(temp_dir,"tempaudio.wav")
    temp = open(wav_path,"wb")
    dec = base64.b64decode(data)
    temp.write(dec)
    temp.close()
    data, sr = librosa.load(wav_path, sr=16000)
    data = nr.reduce_noise(y=data, sr=sr)
    sf.write(os.path.join(temp_dir,"reduced.wav"), data, sr, subtype='PCM_24')
    available_emotions = ["anger", "disgust", "fear", "joy", "sadness"]

    text_emotions = dict()
    for d in predict_text_emotion(data, sr):
        if d["label"] in available_emotions:
            text_emotions[d["label"]] = d["score"]

    speech_emotions = dict()
    for d in  predict_speech_emotion(data, sr):
        speech_emotions[d["label"]] = float(d["score"])
    speech_emotions["joy"] = speech_emotions["happiness"]
    del speech_emotions["happiness"]
    
    return {"text_emotion": text_emotions, "speech_emotion": speech_emotions}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)