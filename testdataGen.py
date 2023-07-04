import numpy as np
import librosa
import pandas as pd

# Function to generate the test audio signal
def generate_audio_signal(bandwidth, freq, boost):
    # Replace this with your own code to generate or load the audio signal
    audio_signal = np.zeros(44100)  # Placeholder audio signal
    return audio_signal

# Function to extract features from the audio signal using Librosa
def extract_features(audio_signal):
    # Replace this with your own code to extract audio features
    features = {}  # Placeholder features dictionary
    features['chroma_stft'] = librosa.feature.chroma_stft(y=audio_signal)
    features['rmse'] = librosa.feature.rms(y=audio_signal)
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio_signal)
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio_signal)
    features['rolloff'] = librosa.feature.spectral_rolloff(y=audio_signal)
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=audio_signal)
    features['mfcc'] = librosa.feature.mfcc(y=audio_signal)
    return features

# Define the settings for the test signals
settings = [
    {'bandwidth': 2, 'freq': 98, 'boost': 8},
    {'bandwidth': 4, 'freq': 530, 'boost': 8}
]

# Initialize an empty DataFrame to store the test data
test_data = pd.DataFrame(columns=['filename', 'chroma_stft', 'rmse', 'spectral_centroid',
                                  'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
                                  'mfcc', 'label'])

# Generate the test data for each setting
for setting in settings:
    bandwidth = setting['bandwidth']
    freq = setting['freq']
    boost = setting['boost']
    
    # Create or load the test audio signal for the current setting
    audio_signal = generate_audio_signal(bandwidth, freq, boost)
    
    # Extract the features from the audio signal
    features = extract_features(audio_signal)
    
    # Create a row for the test data with the extracted features and the setting label
    row = {'filename': 'test_signal.wav', 'label': f'Bandwidth={bandwidth}, Freq={freq}, Boost={boost}'}
    row.update(features)
    
    # Add the row to the test data DataFrame
    # test_data = test_data.loc[len(test_data)] = row
    new_index = len(test_data)
    # Add the row to the DataFrame using .loc
    test_data.loc[new_index] = row  

# Save the test data to a CSV file
test_data.to_csv('test_data.csv', index=False)
