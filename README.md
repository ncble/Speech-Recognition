# Speech-Recognition

## Remove silence from audio extracts

Script `reduce.py` -> not efficient at all !!! (size of data inscreases)

	$ python reduce.py

`librosa.load` is used to load data (problems of normalization with scipy)

## Preprocessing

	$ python preprocessing.py

2 types:
+ mel spectogram -> np array of size (128,21)
+ mfc_coeffs -> np array of size (13, 21)
+ TODO: classic spectogram
