import sys
sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
from draw import draw_clouds
import numpy as np
import umap
from time import time


if __name__=="__main__":
	print("start")

	filename_X = "../data/preprocessed/input.npy"
	filename_Y = "../data/preprocessed/output.npy"

	X = np.load(filename_X).reshape(5500,-1)
	Y = np.argmax(np.load(filename_Y), axis=1)
	# import ipdb; ipdb.set_trace()

	n_neighbors = 10
	min_dist = 0.001
	n_components = 3
	metric = 'correlation' #'manhattan'## "correlation", 'euclidean' (default), "cosine", 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'mahalanobis'
	#, 'mahalanobis', 'wminkowski', 'seuclidean', 'haversine', 'hamming', 'jaccard', 'dice', 'russelrao', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'yule'
	st = time()
	embedding = umap.UMAP(n_neighbors=n_neighbors,
						  min_dist=min_dist,
						  metric=metric,
						  n_components=n_components).fit_transform(X)
	fig_path = "n_neighbors_{}_min_dist_{}_metric_{}.png".format(n_neighbors, min_dist, metric)
	title = "n_neighbors {} min_dist {} metric {}".format(n_neighbors, min_dist, metric)
	draw_clouds(embedding, labels=Y, save_to="../figures/audio_umap_embeddings/"+fig_path, title=title)

	# good: n_neighbors = 10;  min_dist = 0.001; n_components = 3; metric = 'cosine'