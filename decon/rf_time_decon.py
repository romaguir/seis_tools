from rf.deconvolve import _toeplitz_real_sym
import numpy as np

def rf_time_decon(a,b):
   return _toeplitz_real_sym(np.hstack((a,a[:1])),b)
