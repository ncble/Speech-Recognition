# _*_ coding: utf-8 _*_
import os
import numpy as np
import matplotlib.pyplot as plt
import cma

from cma.constraints_handler import BoundTransform
from cma import fitness_transformations as ft

if __name__ == "__main__":

	def draw_surface_level(fun, centre = np.zeros(2), taille = 1.0, message = None):

		plt.figure(figsize=(5, 3))
		axes = plt.gca()
		if message:
			plt.title(message)
		x_min, y_min = (centre - taille)#[0,0], (centre - taille)[0,1]  #X[:, 0].min()
		x_max, y_max = (centre + taille)#[0,0], (centre + taille)[0,1]     #X[:, 0].max()
		
		axes.set_xlim([x_min,x_max])
		axes.set_ylim([y_min,y_max])
		XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

		Z = fun(np.c_[XX.ravel(), YY.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(XX.shape)
		plt.pcolormesh(XX, YY, Z, cmap = plt.cm.jet)
		# CS = plt.contour(XX, YY, Z, cmap = plt.cm.jet)
		# plt.clabel(CS, fmt='%2.1f', colors='b', fontsize=14)
		plt.show()

	def slope(x):

		tengent = -5
		# return np.sum(tengent*x, axis = 1)
		return np.sum(tengent*x)

	# cma.CMAOptions()
	# draw_surface_level(slope)
	# CMA_cls = cma.constraints_handler.BoundaryHandlerBase([np.array([-5,-5]), np.array([5,5])])
	# cma.fmin(cma.ff.sphere, np.ones(2), 0.1, options = {"BoundaryHandler":None})
	
	# bounded_sphere = ft.ComposedFunction([cma.ff.sphere, BoundTransform([[], 5 * [-1] + [np.inf]]).transform]) # 
	res = cma.fmin(slope, np.ones(2), 0.5, options={'bounds': [[-5,-5], [5,5]]}) #'BoundaryHandler': cma.s.ch.BoundTransform, 
	cma.plot()
	plt.savefig("./test_cma_plot.png")
	print(res[0])
	# import ipdb; ipdb.set_trace()

	# print CMA_cls.get_bounds('lower', 2)
	# print CMA_cls.get_bounds('upper', 2)
	# 