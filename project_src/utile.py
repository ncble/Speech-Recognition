# _*_ coding: utf-8 _*_

import os

"""
Original files/folders management


"""

def get_dir_name(full_path, return_prefix=False):
	"""
	Take full FOLDER path as input, and return the directory name 

	(won't work is the path contains filepath like  "./dirname/file.txt")

	Ps.
	The command 'os.path.basename' is not consistent in different version of python.
	That's why I use 'split' here.

	"""
	# folder_name = os.path.basename(full_folder_path) #"./test_folder/1/2/3" => 3, "./test_folder/1/2/3/" => ""

	folder_name = full_path.split("/")[-1]
	prefix = "/".join(full_path.split("/")[:-1])
	# print(full_path.split("/"))
	if len(folder_name) == 0:
		print("Warning: Folder path should look like '/XXX/Name' not '/XXX/Name/'.")
		folder_name = full_path.split("/")[-2]
		prefix = "/".join(full_path.split("/")[:-2])

	if not return_prefix:
		return folder_name
	else:
		return prefix, folder_name

def makedirs_advanced(full_folder_path, set_count=0):
	
	"""
	Make new directory in an intelligent way (for the research purpose). If you want to create folder "./data/test" and it already exists,
	then it will create folder "./data/test_1", "./data/test_2" ... and so on by analysing folder index.
	
	Please set_count= k to whatever integer you want to start.


	return : makedirs and return the full path
	"""

	
	prefix, folder_name = get_dir_name(full_folder_path, return_prefix=True)
	count = set_count
	
	current_folder_path = os.path.join(prefix, folder_name+"_{}".format(count))

	while os.path.exists(current_folder_path):
		count += 1
		current_folder_name = folder_name+"_{}".format(count)
		current_folder_path = os.path.join(prefix, current_folder_name)
	os.makedirs(current_folder_path)

	return current_folder_path

def filetype(filename):
	return filename.split("/")[-1].split(".")[-1]

def generator(root_dir='../data/', file_type=None, dir_label_fun=None, file_label_fun=None, stop_after = None, subfolder = None, verbose=False):
	"""
	A simple generator based on os.walk. An additional function "dir_label_fun" could be use to 
	return the name of class of file. (When doing classification task in Machine Learning)
	
	Input:
		file_type: A filter by filename extension. Default to None.
		dir_label_fun: A function taking str as input and output a str. Default to None.
		file_label_fun: A function taking str as input and output a str. Default to None.
		verbose: print the folder structure. Default to False.
		stop_after: int, stop after. Default to 0.
		subfolder: Specify the searching subfolder name. (maybe useless?) Default to None.

	See for more details: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	
	yield: full file path

	"""
	assert (dir_label_fun is None) or (file_label_fun is None), "Not support two(multi-) labels yet."

	if subfolder is not None: # == if subfolder:
		root_dir = os.path.join(root_dir, subfolder)
	count = 0
	for path, directories, files in os.walk(root_dir):
		# print 'ls %r' % path
		if verbose:
			print('ls %r' % path)
			for directory in directories:
				print("    d'{}'".format(directory))
				# print('Reading directory {}...'.format(directory))

		for filename in files:
			if (stop_after is not None) and count >= stop_after:
				break
			if (file_type is not None) and (filetype(filename) != file_type):
				continue
			if verbose:
				print('    -%r' % filename)

			if dir_label_fun is not None:
				dir_name = get_dir_name(path, return_prefix=False)
				classe_label = dir_label_fun(dir_name)

				yield os.path.join(path, filename), classe_label

			elif file_label_fun is not None:
				classe_label = file_label_fun(filename)

				yield os.path.join(path, filename), classe_label
			else:

				yield os.path.join(path, filename)
			count += 1
	if verbose:
		print("Total files number {}".format(count))


### Old version here ###
# def generator(root_dir='../data/train/audio/', folder = None, stop_after = None):
# 	"""
# 	A simple generator. See: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
#	
# 	"""
# 	if folder is not None: # == if folder:
# 		root_dir = os.path.join(root_dir, folder)
# 	count = 0
# 	for path, directories, files in os.walk(root_dir):
# 		# print 'ls %r' % path
# 		# for directory in directories:
# 		# 	print('Reading directory {}...'.format(directory))
# 		for filename in files:
# 			if (stop_after is not None) and count >= stop_after:
# 				break
# 			print '    -%r' % filename
# 			yield os.path.join(path, filename)
# 			count += 1


def resize_audio(y, sr, new_sr=8000.):
	assert type(new_sr) == float, "new_sr need to be a float (Python2)"
	new_y = signal.resample(y, int(new_sr/sr * len(y)))

	return new_y, new_sr



if __name__ == "__main__":
	print("Start...")



	# y, sr = example()
	# plot_fourier_librosa(y, sr)
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


	
	# import ipdb; ipdb.set_trace()

	# spectre = plot_fourier_librosa(y, sr)
	# import ipdb; ipdb.set_trace()


	# walker_example()

	# for i in generator(folder = "no", stop_after = 10):
	# 	print(i)
		


