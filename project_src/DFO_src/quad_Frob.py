"""
Description:
An implementation of the DFO algorithm developed by A. Conn,
K. Scheinberg, L. Vicente
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
import numpy as np
from scipy import linalg as LA


# Packages Lu
# from keras.models import Sequential
# from keras.layers import Dense 
# from keras import optimizers

# sgd = optimizers.SGD(lr=0.00001)
#############################################################

def _compute_coeffs(W, tol_svd, b, option):
	"""Compute model coefficients -- Here we are merely solving the
	 system of equations -- decompose W and use its inverse
	"""
	if option == 'partial':
		U, S, VT = LA.svd(W)
	else:
		U, S, VT = LA.svd(W, full_matrices=False)

	# Make sure the condition number is not too high (!!! "low" Lu)
	indices = S < tol_svd
	S[indices] = tol_svd
	Sinv = np.diag(1/S)
	V = VT.T
	# Get the coefficients
	lambda_0 = reduce(np.dot, [V, Sinv, U.T, b])
	return (lambda_0)

def preprocessing(b):
	# We suppose that b is 1-D array
	moy = np.mean(b)
	std = np.std(b)
	new_b = (b-moy)/std
	return new_b, moy, std


def linear_reg_model(N, optimizer='sgd'):
	model = Sequential()
	model.add(Dense(1, input_shape=(N,))) 
	model.compile(loss='mean_squared_error', optimizer=optimizer) # 'rmsprop'
	return model

def _compute_coeffs_lu(W, tol_svd, b, option): # tol_svd and option are useless

	m, N = W.shape  # N = (dim+1)(dim+2)/2  b.shape = (m,1)  # m = number of points
	model = linear_reg_model(N)
	model.fit(W, b, epochs=700, shuffle=True, batch_size= 128)
	lambda_from_grad = model.layers[0].get_weights()[0] # W_Output_Hidden
	
	return np.vstack((np.ones(1), lambda_from_grad)) # bias is not important

def _compute_coeffs_lu2(W, tol_svd, b, option, epochs = 50, set_weights=None, optimizer='sgd'):
	# Remove the first column (of 1)
	W = W[:,1:] # [IM]
	####
	# import ipdb; ipdb.set_trace()
	m, N = W.shape  # N = (dim+1)(dim+2)/2 -1  b.shape = (m,1)  # m = number of points 
	
	model = linear_reg_model(N, optimizer=optimizer)
	if set_weights is not None:
		# print("Setting pretrained weights...")
		model.layers[0].set_weights(set_weights)

	model.fit(W, b, epochs=epochs, shuffle=True, batch_size = 128)

	weights = model.layers[0].get_weights()[0]
	bias = model.layers[0].get_weights()[1]
	return np.vstack((bias, weights)) 

	
def quad_Frob(X, F_values):
	"""
	Given a set of points in the trust region
	and their values, construct a quadratic model
	in the form of g.T x + 1/2 x.T H x + \alpha.

	If the number of points are less than
	(n+1) (n+2)/2 then build the model such that the
	Frobenius norm is minimized. In this code the KKT
	conditions are solved. Otherwise, solve
	the system of equations used in polynomial interpolation.
	M(\phi, Y) \lambda = f

	arguments: X: the sample points to be interpolated
		F_values: the corresponding true solutions to the sample points
	outputs: g and H in the quadratic model
	"""
	# Minimum value accepted for a singular value
	eps = np.finfo(float).eps
	tol_svd = eps**5
	# n = number of variables m = number of points
	(n, m) = X.shape

	H = np.zeros((n, n))
	g = np.zeros((n, 1))

	# Shift the points to the origin
	Y = X - np.dot(np.diag(X[:, 0]), np.ones((n, m)))

	if (m < (n+1)*(n+2)/2):
		# Construct a quad model by minimizing the Frobenius norm of
		# the Hessian -- the following is the solution of the KKT conditions
		# of the optimization problem on page 81 of the Intro to DFO book
		b = np.vstack((F_values, np.zeros((n+1, 1))))
		A = 0.5 * (np.dot(Y.T, Y)**2)

		# Construct W by augmenting the vector of ones with the linear and
		# quadratic terms. The first m rows build the matrix M, which is
		# introduced in the slides (monomials of quadratic basis)
		top = np.hstack((A, np.ones((m, 1)), Y.T))
		temp = np.vstack((np.ones((1, m)), Y))
		bottom = np.hstack((temp, np.zeros((n+1, n+1))))
		W = np.vstack((top, bottom))
		
		# b_scaled, moy, std = preprocessing(b) # Added by Lu [IM]
		lambda_0 = _compute_coeffs(W, tol_svd, b, option='partial') # Commented by Lu 
		# lambda_0 = _compute_coeffs(W, tol_svd, b_scaled, option='partial') # Modified by Lu
		# lambda_0 = _compute_coeffs_lu(W, tol_svd, b, option='partial') # Added by Lu
		# lambda_0 *= std # Added by Lu [IM]
		
		# Grab the coeffs of linear terms (g) and the ones of quadratic terms
		# (H) for g.T s + s.T H s
		g = lambda_0[m+1:]

		H = np.zeros((n, n))
		for j in range(m):
			H = H + (lambda_0[j] *
					np.dot(Y[:, j].reshape(n, 1), Y[:, j].reshape(1, n)))

	else:  # Construct a full model
		# Here we have enough points. Solve the sys of equations.
		b = F_values
		# np.savetxt(open("./b.txt", "wb"), b, delimiter = " ") # Added by Lu
		

		phi_Q = np.array([])
		for i in range(m):
			y = Y[:, i]
			y = y[np.newaxis]  # turn y from 1D to a 2D array
			aux_H = y * y.T - 0.5 * np.diag(pow(y, 2)[0])
			aux = np.array([])
			for j in range(n):
				aux = np.hstack((aux, aux_H[j:n, j]))

			phi_Q = np.vstack((phi_Q, aux)) if phi_Q.size else aux

		W = np.hstack((np.ones((m, 1)), Y.T))
		W = np.hstack((W, phi_Q))
		# np.savetxt(open("./W.txt", "wb"), W, delimiter = " ") # Added by Lu
		# b_scaled, moy, std = preprocessing(b) # Added by Lu [IM]
		lambda_0 = _compute_coeffs(W, tol_svd, b, option='full') # Commented by Lu
		# lambda_0 = _compute_coeffs(W, tol_svd, b_scaled, option='full') # Modified by Lu
		# lambda_0 = _compute_coeffs_lu(W, tol_svd, b, option='full') # Added by Lu
		# lambda_0 = _compute_coeffs_lu(W, tol_svd, b_scaled, option='full') # Added by Lu
		# lambda_0 = _compute_coeffs_lu2(W, tol_svd, b_scaled, option='full')

		###
		# lambda_from_svd_temp = _compute_coeffs(W, tol_svd, b_scaled, option='full')
		# pretrained_weights = [lambda_from_svd_temp[1:].reshape(-1,1), np.array(lambda_from_svd_temp[0]).reshape(1,)]
		# lambda_0 = _compute_coeffs_lu2(W, tol_svd, b_scaled, option='full', epochs = 50, set_weights = pretrained_weights, optimizer = sgd)
		###
		
		# lambda_0 *= std # Added by Lu [IM]

		# import ipdb; ipdb.set_trace()
		# Retrieve the model coeffs (g) and (H)
		g = lambda_0[1:n+1, :]
		cont = n+1
		H = np.zeros((n, n))

		for j in range(n):
			H[j:n, j] = lambda_0[cont:cont + n - j, :].reshape((n-j,))
			cont = cont + n - j

		H = H + H.T - np.diag(np.diag(H))
	return (H, g)

