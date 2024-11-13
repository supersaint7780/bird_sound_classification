import streamlit as st
import numpy as np 
import tensorflow as tf
import librosa
from collections import Counter

def most_frequent(arr):
    """
    Returns the most frequent value in the array.
    If there is a tie, returns one of the most frequent values.
    """
    frequency_dict = {}
    most_frequent_element = None
    highest_frequency = 0

    # Count occurrences of each element
    for element in arr:
        if element in frequency_dict:
            frequency_dict[element] += 1
        else:
            frequency_dict[element] = 1

        # Update most frequent element if current element has higher frequency
        if frequency_dict[element] > highest_frequency:
            most_frequent_element = element
            highest_frequency = frequency_dict[element]

    return most_frequent_element


def load_and_preprocess_data(audio, target_shape = (128, 313), rms_threshold=0.01, n_fft_value=2048):
    data = []
                
    # Load audio
    audio_data, sample_rate = librosa.load(audio, sr=None)
                
    chunk_duration = 5  # in seconds
    overlap_duration = 2  # in seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
                            
        # Calculate RMS to filter out noise
        rms = np.sqrt(np.mean(chunk**2))
        if rms < rms_threshold:
            continue  # Skip this chunk as it's likely noise

        # If chunk is shorter than n_fft, pad it with zeros
        if len(chunk) < n_fft_value:
            chunk = np.pad(chunk, (0, n_fft_value - len(chunk)))
                    
        # Compute mel spectrogram with dynamic n_fft
        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk, sr=sample_rate, n_fft=min(n_fft_value, len(chunk))
        )
                    
        # Resize or pad the mel spectrogram to match target shape
        mel_spectrogram_resized = np.resize(np.expand_dims(mel_spectrogram, axis=1), target_shape)
                    
        # Append to data and labels
        data.append(mel_spectrogram_resized)
    
    return np.array(data)


MODEL_PATH = "./final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("CNN-Based Bird Sound Classification")

uploaded_file = st.file_uploader("Upload a sound file", type=[ "ogg", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/ogg')
    
    data = load_and_preprocess_data(uploaded_file)
    data_shape = data.shape
    data = data.reshape(data_shape[0], data_shape[1], data_shape[2], 1)

    pred = model.predict(data)
    y_pred = np.argmax(pred, axis = 1)
    print(y_pred)
    
    classes = [
        'Barn Swallow',
        'collared dove',
        'gray heron',
        'green sandpiper',
        'eurasian hoopoe',
        'zitting cisticola'
    ]

    most_frequent_class = classes[most_frequent(y_pred)]

    # most_frequent_class = 1
    # Display the final prediction result
    st.write(f"The voice is of : {most_frequent_class}")

