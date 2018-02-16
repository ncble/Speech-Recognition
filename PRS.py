# Pure random search
import numpy as np

def PRS(f, dim, n_eval, bounds = [0., 1.]):
	x_opt, y_opt = None, np.infty
	for k in range(n_eval):
		x = np.random.uniform(low = bounds[0], high = bounds[1], size = dim)
		f_x = f(x)
		if f_x<y_opt:
			x_opt, y_opt = x, f_x
		n_eval += 1
	return x_opt, y_opt
