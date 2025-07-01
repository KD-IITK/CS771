import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0

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


################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
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


################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
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