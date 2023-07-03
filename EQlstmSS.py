import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Flatten, Dense
from keras.utils import to_categorical
from scipy.signal import chirp

# Generate a sine sweep test signal
duration = 5  # Duration of the sine sweep in seconds
sampling_rate = 44100  # Sampling rate of the audio signal

# Create a time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the sine sweep signal
frequency_start = 20  # Starting frequency of the sweep
frequency_end = 20000  # Ending frequency of the sweep
sine_sweep = chirp(t, f0=np.log10(frequency_start), f1=np.log10(frequency_end), t1=duration, method='logarithmic')
# Convert back to linear frequency scale
sine_sweep = 10 ** sine_sweep

# Plot the sine sweep signal
plt.figure(figsize=(12, 4))
plt.plot(t, sine_sweep)
plt.title('Sine Sweep Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Load regular signals from data.csv
data = pd.read_csv('data.csv')
regular_signals = []
for filename in data['filename']:
    try:
        signal, sr = librosa.load(filename)
        regular_signals.append(signal)
    except FileNotFoundError:
        print(f"File not found: {filename}")

regular_labels = data['label']

for file_name in data['filename']:
    signal, sr = librosa.load(file_name)
    regular_signals.append(signal)


test_labels = [1]  # Label for the test signal

# Combine regular and test signals
all_signals = regular_signals + [sine_sweep]
all_labels = regular_labels.tolist() + test_labels

# Convert signals to spectrograms
spectrograms = []
for signal in all_signals:
    spectrogram = librosa.feature.melspectrogram(signal, sr=sr)
    spectrograms.append(spectrogram)

# Convert spectrograms to 3D arrays
X = np.array(spectrograms)
y = np.array(all_labels)

# Split the dataset into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Reshape the input for LSTM
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], train_X.shape[2], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))

# Convert labels to categorical
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# Build and train the model with LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2], 1)))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#model.fit(train_X, train_y, batch_size=64, epochs=100, validation_data=(test_X, test_y))
