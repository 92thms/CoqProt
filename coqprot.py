import os
import sys
import configparser
import logging
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
from scipy.io.wavfile import write
from datetime import datetime
from flask import Flask, render_template, send_from_directory, Response
import threading
import queue

# Konfiguration laden
config = configparser.ConfigParser()
config.read('config.ini')

# Logger einrichten
log_file = config['Logging']['log_file']
log_level = getattr(logging, config['Logging']['log_level'].upper(), logging.INFO)
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s:%(levelname)s:%(message)s')

# Audioeinstellungen
SAMPLE_RATE = int(config['Audio']['sample_rate'])
CHUNK_SIZE = int(config['Audio']['chunk_size'])
CHANNELS = int(config['Audio']['channels'])
DEVICE_INDEX = config['Audio']['device_index']
if DEVICE_INDEX == 'None':
    DEVICE_INDEX = None
else:
    DEVICE_INDEX = int(DEVICE_INDEX)

# Erkennungseinstellungen
THRESHOLD = float(config['Detection']['threshold'])
MODEL_PATH = config['Detection']['model_path']

# Aufnahmeeinstellungen
CLIP_DURATION = int(config['Recording']['clip_duration'])
SAVE_PATH = config['Recording']['save_path']
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Webinterface-Einstellungen
HOST = config['WebInterface']['host']
PORT = int(config['WebInterface']['port'])
DEBUG = config['WebInterface'].getboolean('debug')

# TensorFlow Lite Modell laden
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Flask-App erstellen
app = Flask(__name__)

# Audio-Queue für Live-Stream
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        logging.warning(f'Audio-Status: {status}')
    # Audio-Daten für die Erkennung kopieren
    audio_data = indata.copy().flatten()
    # In die Audio-Queue für den Live-Stream legen
    audio_queue.put(audio_data.tobytes())
    # Hahnenkrähen-Erkennung
    if detect_rooster(audio_data):
        save_audio_clip(audio_data)

def detect_rooster(audio_data):
    # Merkmalsextraktion (MFCCs)
    mfccs = np.mean(np.abs(np.fft.rfft(audio_data))[:50])
    # Vorbereitung für das Modell
    input_data = np.array([[mfccs]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]
    if prediction > THRESHOLD:
        logging.info('Hahnenkrähen erkannt')
        return True
    return False

def save_audio_clip(audio_data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rooster_{timestamp}.wav'
    filepath = os.path.join(SAVE_PATH, filename)
    write(filepath, SAMPLE_RATE, audio_data)
    logging.info(f'Audioclip gespeichert: {filepath}')

def start_audio_stream():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE, device=DEVICE_INDEX):
        threading.Event().wait()

@app.route('/')
def index():
    recordings = sorted(os.listdir(SAVE_PATH), reverse=True)
    return render_template('index.html', recordings=recordings)

@app.route('/recordings/<filename>')
def download_recording(filename):
    return send_from_directory(SAVE_PATH, filename)

def generate_audio_stream():
    while True:
        data = audio_queue.get()
        yield data

@app.route('/audio_feed')
def audio_feed():
    return Response(generate_audio_stream(), mimetype="audio/wav")

def start_web_interface():
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)

if __name__ == '__main__':
    # Threads für Audioaufnahme und Webinterface starten
    audio_thread = threading.Thread(target=start_audio_stream)
    web_thread = threading.Thread(target=start_web_interface)
    audio_thread.start()
    web_thread.start()
    audio_thread.join()
    web_thread.join()
