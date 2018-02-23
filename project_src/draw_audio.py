# _*_ coding: utf-8 _*_
# source activate audio

##### Package necessary #####
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
#############################


##### Package optional #####
 
# For 3D spectrogram plot # Attention: only work within jupyter notebook
# import plotly.graph_objs as go
# import plotly.offline as py
#############################

def example():
	filename = librosa.util.example_audio_file()
	y, sr = librosa.load(filename)
	return y, sr

def example2(T = 10.0, sr = 22050, freq_list = [600,400], coeffs_amplitude= None):
	"""
	Combination of two sinusoidal signals.
	
	T : length of sound
	
	"""
	final_x = 0
	t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
	if coeffs_amplitude is None:
		coeffs_amplitude = np.ones(len(freq_list))

	for index,freq in enumerate(freq_list):
		x = coeffs_amplitude[index]*np.sin(2*np.pi*freq*t) # pure sine wave at 'freq' Hz
		final_x += x
	
	return final_x, sr

def example3(T = 10.0, sr = 22050, freq_list = [600,400]):
	"""
	Inverse signal of example2(). 

	Demonstration of destructive wave: example2()+example3() ~= 0

	"""
	assert len(freq_list) == 2, "The sum of two sinusoidal functions."
	freq = freq_list[0]
	freq2 = freq_list[1]
	t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
	z = -2.0*np.sin(np.pi*(freq+freq2)*t)*np.cos(np.pi*(freq-freq2)*t)

	return z, sr

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
	elif mode == "scipy":
		freqs, times, spectrogram = log_specgram(y, sr)
		plt.figure(figsize=(12, 5))
		librosa.display.specshow(spectrogram.T, sr=sr, x_axis='time', y_axis='hz')
		plt.show()
	else:
		raise ValueError("mode should be 'librosa' or 'scipy'.")

	

def plot_spectrogram_3D(y, sr):
	# TODO
	# X = librosa.stft(y)
	# Xdb = librosa.amplitude_to_db(X)
	freqs, times, spectrogram = log_specgram(y, sr)
	data = [go.Surface(z=spectrogram.T)]
	layout = go.Layout(
		title='Specgtrogram of "yes" in 3d',
		scene = dict(
		yaxis = dict(title='Frequencies', range=freqs),
		xaxis = dict(title='Time', range=times),
		zaxis = dict(title='Log amplitude'),
		),)


	fig = go.Figure(data=data, layout=layout)
	py.iplot(fig)
	py.show()


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
	# assert 1==2, "Not finished yey !"
	T = 1.0 / sr
	N = y.shape[0]
	xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	spectre = np.abs(librosa.stft(y))
	# import ipdb; ipdb.set_trace()


	vals = 2.0/N * spectre[0:N//2]

	plt.figure(figsize=(12, 4))
	plt.title('FFT of recording sampled with ' + str(sr) + ' Hz')
	plt.plot(xf, vals)
	plt.xlabel('Frequency')
	plt.grid()
	plt.show()
	return xf, vals

	# return spectre


def plot_Mel_spectrogram(y, sr):
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.power_to_db(S, ref=np.max)

	plt.figure(figsize=(12, 4))
	librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
	plt.title('Mel power spectrogram ')
	plt.colorbar(format='%+02.0f dB')
	plt.tight_layout()
	plt.show()
	return

def plot_MFCC_coeffs(y, sr):
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128) # shape = (128, ???)
	log_S = librosa.power_to_db(S, ref=np.max)

	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13) # shape = (13, ???)

	delta2_mfcc = librosa.feature.delta(mfcc, order=2) # shape = (13, ???)

	# import ipdb; ipdb.set_trace()
	plt.figure(figsize=(12, 4))
	librosa.display.specshow(delta2_mfcc)
	plt.ylabel('MFCC coeffs')
	plt.xlabel('Time')
	plt.title('MFCC')
	plt.colorbar()
	plt.tight_layout()
	plt.show()
	return

def log_specgram(audio, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	print("Warning... I don't understand this yet.")
	print("Warning... I don't understand this yet.")
	print("Warning... I don't understand this yet.")

	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,
									fs=sample_rate,
									window='hann',
									nperseg=nperseg,
									noverlap=noverlap,
									detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def load_audio_file(filename):
	# y, sr = librosa.load(filename, sr=22050, offset=0.0, duration=None)
	sr, y = wavfile.read(filename)
	# y, sr = librosa.core.load(filename, sr=22050, offset=15.0, duration=5.0)
	return y, sr

