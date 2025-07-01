import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao

def my_fit(X_train, y_train):
	X_train_mapped = my_map(X_train)
	model = LinearSVC(
		loss='hinge',         
		C=1.0,                
		tol=1e-4,             
		max_iter=10000,       
		random_state=42       
	)
	model.fit(X_train_mapped, y_train)
	w = model.coef_.flatten()
	b = model.intercept_[0]

	return w, b

def my_map(X):
	X = 2 * X - 1
	n_samples, n_bits = X.shape
	phi = np.zeros((n_samples, n_bits + 1))
	phi[:, 0] = 1  # Bias term

	for i in range(n_bits):
		phi[:, i + 1] = phi[:, i] * X[:, i]

	feat = np.zeros((n_samples, (n_bits + 1) * (n_bits + 1)))
	for i in range(n_samples):
		feat[i] = np.kron(phi[i], phi[i])

	return feat

def my_decode(w):
	weights = w[:-1]
	bias = w[-1]

	n = 64
	p = np.zeros(n)
	q = np.zeros(n)
	r = np.zeros(n)
	s = np.zeros(n)
	
	beta = np.zeros(n)
	alpha = np.zeros(n)
	
	beta[-1] = bias
	
	for i in range(n-1, 0, -1):
		alpha[i] = weights[i] - beta[i-1]
	
	alpha[0] = weights[0]
	
	for i in range(n):

		p_i = alpha[i] + beta[i]
		q_i = 0
		r_i = 0
		s_i = alpha[i] - beta[i]
		
		if s_i < 0:
			s_i = 0
			p_i = 2 * alpha[i]
		
		p[i] = max(p_i, 0)
		q[i] = max(q_i, 0)
		r[i] = max(r_i, 0)
		s[i] = max(s_i, 0)

	return p, q, r, s