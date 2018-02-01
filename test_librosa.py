# _*_ coding: utf-8 _*_
# source activate py3

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
# Beat tracking example
# import IPython.display as ipd


# 1. Get the file path to the included audio example
filename = librosa.util.example_audio_file()
print()
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)


import ipdb; ipdb.set_trace()
# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(y, sr=sr)
# plt.show()
