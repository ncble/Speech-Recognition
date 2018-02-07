# _*_ coding: utf-8 _*_
# source activate audio

##### Package necessary #####
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile
from scipy.fftpack import fft
#############################


def example():
	filename = librosa.util.example_audio_file()
	y, sr = librosa.load(filename)
	return y, sr

def cut(y, threshold = 10):
	# Silence removal
	y_cut, index = librosa.effects.trim(y, top_db = threshold)
	print(index)
	y[:index[0]] = 0
	y[index[1]:] = 0
	return y
	# return y_cut

def plot_wave(y, sr):
	plt.figure(figsize=(12, 4))
	librosa.display.waveplot(y, sr=sr)
	plt.show()

def plot_spectrogram(y, sr, mode = "librosa"):
	
	if mode == "librosa":
		X = librosa.stft(y)
		Xdb = librosa.amplitude_to_db(X)
		plt.figure(figsize=(12, 5))
		librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
		plt.show()
	return log_S

def custom_fft(y, fs):
	T = 1.0 / fs
	N = y.shape[0]
	yf = fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
	# FFT is also complex, to we take just the real part (abs) 
	# (Lu: J'ai pas très bien compris cela.)
	return xf, vals

def plot_fourier(y, sr):
	xf, vals = custom_fft(y, sr)
	plt.figure(figsize=(12, 4))
	plt.title('FFT of recording sampled with ' + str(sr) + ' Hz')
	plt.plot(xf, vals)
	plt.xlabel('Frequency')
	plt.grid()
	plt.show()

def plot_fourier_librosa(y, sr):
	assert 1==2, "Not finished yey !"
	T = 1.0 / sr
	N = y.shape[0]
	xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	spectre = librosa.stft(y).real
	import ipdb; ipdb.set_trace()


	vals = 2.0/N * spectre[0:N//2]

	plt.figure(figsize=(12, 4))
	plt.title('FFT of recording sampled with ' + str(sr) + ' Hz')
	plt.plot(xf, vals)
	plt.xlabel('Frequency')
	plt.grid()
	plt.show()
	return xf, vals

	# return spectre


if __name__ == "__main__":
	print("Start...")



	# y, sr = example()
	# y, sr = example2(freq_list = [600, 400], coeffs_amplitude= [1., 10.])
	# plot_fourier(y,sr)
	# plot_wave(y, sr)
	# plot_spectrogram(y, sr, mode = 2)
	# plot_MFCC_spectrogram(y, sr)
	# plot_MFCC_coeffs(y, sr)
	# plot_spectrogram_3D(y, sr)
	# y, sr = load_audio_file("./music.mp4")
	# y, sr = load_audio_file("./data/train/audio/tree/0ac15fe9_nohash_0.wav")
	# sample_rate, samples = wavfile.read("./data/train/audio/tree/0ac15fe9_nohash_0.wav")


	
	# spectre = plot_fourier_librosa(y, sr)
	# import ipdb; ipdb.set_trace()

