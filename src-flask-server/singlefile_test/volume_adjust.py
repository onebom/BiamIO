import simpleaudio as sa
import scipy.io.wavfile as wav
import numpy as np

bgm_path = '../static/bgm/main.wav'

# Load the audio data from a WAV file
sample_rate, audio_data = wav.read(bgm_path)

# Define a function to adjust the volume
def adjust_volume(audio_data, volume):
  # Scale the amplitude of the audio data by the volume factor
  scaled_audio_data = audio_data * volume
  # Clip the audio data to the range of the data type
  clipped_audio_data = np.clip(scaled_audio_data, -32768, 32767)
  return clipped_audio_data

# Adjust the volume of the audio data
volume = 0.1 # reduce volume by 50%
modified_audio_data = adjust_volume(audio_data, volume)

# Convert the audio data to the required data type (int16) and play it
modified_audio_data = modified_audio_data.astype('int16')
play_obj = sa.play_buffer(modified_audio_data, num_channels=audio_data.shape[1], bytes_per_sample=audio_data.dtype.itemsize, sample_rate=sample_rate)

# Wait for the audio to finish playing
play_obj.wait_done()
