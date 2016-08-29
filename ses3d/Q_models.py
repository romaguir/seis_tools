import numpy as np
import matplotlib.pylab as plt

###################################################################################################
#- plot a Q model
###################################################################################################

def plot_q(model='cem', r_min=0.0, r_max=6371.0, dr=1.0):

	"""
	Plot a radiallysymmetric Q model.

	plot_q(model='cem', r_min=0.0, r_max=6371.0, dr=1.0):

	r_min=minimum radius [km], r_max=maximum radius [km], dr=radius increment [km]

	Currently available models (model): cem, prem, ql6
	"""

	r = np.arange(r_min, r_max+dr, dr)
	q = np.zeros(len(r))

	for k in range(len(r)):

		if model=='cem':
			q[k]=q_cem(r[k])
		elif model=='ql6':
			q[k]=q_ql6(r[k])
		elif model=='prem':
			q[k]=q_prem(r[k])


	plt.plot(r,q,'k')
	plt.xlim((0.0,r_max))
	plt.xlabel('radius [km]')
	plt.ylabel('Q')
	plt.show()


###################################################################################################
#- CEM, EUMOD
###################################################################################################

def q_cem(r):

	"""
	This is the 1D Q model used in the Comprehensive Earth Model. It is a smoothed version of QL6, presented by 
	Durek & Ekstrom (BSSA, 1996).
	"""

	a=(6371.0-r)/271.0

	if ((r<=6371.0) & (r>=6100.0)):

		q=300.0-5370.82*a**2+14401.62*a**3-13365.78*a**4+4199.98*a**5

	elif ((r<=6100.0) & (r>=5701.0)):

		q=165.0

	elif ((r<=5701.0) & (r>=3480.0)):

		q=355.0

	elif ((r<=3480.0) & (r>=1221.0)):

		q=0.0

	else:

		q=104.0

	return q


###################################################################################################
#- QL6
###################################################################################################

def q_ql6(r):

	"""
	This is QL6 by Durek & Ekstrom (BSSA, 1996).
	"""

	if (r>=6346.6):

		q=300.0

	elif ((r<=6346.6) & (r>=6291.0)):

		q=191.0

	elif((r<=6291.0) & (r>=6151.0)):

		q=70.0

	elif ((r<=6151.0) & (r>=5701.0)):

		q=165.0

	elif ((r<=5701.0) & (r>=3480.0)):

		q=355.0

	elif ((r<=3480.0) & (r>=1221.0)):

		q=0.0

	else:

		q=104.0

	return q


###################################################################################################
#- PREM
###################################################################################################

def q_prem(r):

	"""
	This is PREM by Dziewonski & Andreson (PEPI, 1981).
	"""

	if ((r<=6371.0) & (r>=6356.0)):

		q=600.0	

	elif ((r<=6356.0) & (r>=6346.6)):

		q=600.0	

	elif ((r<=6346.6) & (r>=6291.0)):

		q=600.0

	elif ((r<=6291.0) & (r>=6151.0)):

		q=80.0	
	
	elif((r<=6151.0) & (r>=5971.0)):

		q=143.0

	elif ((r<=5971.0) & (r>=5771.0)):

		q=143.0

	elif ((r<=5771.0) & (r>=5701.0)):

		q=143.0

	elif ((r<=5701.0) & (r>=5600.0)):

		q=312.0

	elif ((r<=5600.0) & (r>=3630.0)):

		q=312.0

	elif ((r<=3630.0) & (r>=3480.0)):

		q=312.0
	
	elif ((r<=3480.0) & (r>=1221.5)):

		q=0.0

	elif (r<=1221.5):

		q=84.6

	return q
