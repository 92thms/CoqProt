import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import librosa

# Datenpfade
DATA_DIR = 'data'
ROOSTER_DIR = os.path.join(DATA_DIR, 'rooster')
BACKGROUND_DIR = os.path.join(DATA_DIR, 'background')

# Daten vorbereiten
def load_data(directory, label):
    data = []
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        audio, sr = librosa.load(filepath, sr=22050)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
        data.append((mfccs, label))
    return data

rooster_data = load_data(ROOSTER_DIR, 1)
background_data = load_data(BACKGROUND_DIR, 0)

# Daten kombinieren und mischen
all_data = rooster_data + background_data
np.random.shuffle(all_data)

# Daten aufteilen
X = np.array([x[0] for x in all_data])
y = np.array([x[1] for x in all_data])

# Modell erstellen
model = tf.keras.models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(13,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modell trainieren
model.fit(X, y, epochs=50, batch_size=16)

# Modell speichern
model.save('models/rooster_model.h5')

# In TensorFlow Lite konvertieren
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/rooster_model.tflite', 'wb') as f:
    f.write(tflite_model)
