import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse.linalg import lsmr
def damped_lstsq(a,b,damping=1.0,plot=False):
    '''
    Gm = d
       G : discrete convolution matrix
       m : signal we are trying to recover (receiver function)
       d : the convolved data (signal a)

       m = (G^T G)^(-1)G^T * d
    '''

    #build G
    padding = np.zeros(a.shape[0] - 1, a.dtype)
    first_col = np.r_[a, padding]
    first_row = np.r_[a[0], padding]
    G = toeplitz(first_col, first_row)

    #reshape b
    shape = G.shape
    shape = shape[0]
    len_b = len(b)
    zeros = np.zeros((shape-len_b))
    b = np.append(b,zeros)

    #solve with scipy.sparse.linalg.lsmr
    sol = lsmr(G,b,damp=damping)
    m_est = sol[0]
    rf = m_est

    if plot==True:
       fig,axes = plt.subplots(3,sharex=True)
       axes[0].plot(a)
       axes[1].plot(b)
       axes[2].plot(rf)
       plt.show()

    return rf

