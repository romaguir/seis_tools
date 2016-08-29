import numpy as np
import matplotlib.pylab as plt
import obspy.signal.filter as flt


def make_stf(dt=0.10, nt=5000, fmin=1.0/100.0, fmax=1.0/8.0, filename='../INPUT/stf_new', plot=True, get_stf=False):

	"""
	Generate a source time function for ses3d by applying a bandpass filter to a Heaviside function.

	make_stf(dt=0.13, nt=4000, fmin=1.0/100.0, fmax=1.0/8.0, filename='../INPUT/stf_new', plot=True)

	dt: Length of the time step. Must equal dt in the event_* file.
	nt: Number of time steps. Must equal to or greater than nt in the event_* file.
	fmin: Minimum frequency of the bandpass.
	fmax: Maximum frequency of the bandpass.
	filename: Output filename.

	"""

	#- Make time axis and original Heaviside function. --------------------------------------------

	t = np.arange(0.0,float(nt+1)*dt,dt)
	h = np.ones(len(t))

	#- Apply filters. -----------------------------------------------------------------------------

	h = flt.highpass(h, fmin, 1.0/dt, 3, zerophase=False)
	h = flt.lowpass(h, fmax, 1.0/dt, 5, zerophase=False)

	#- Plot output. -------------------------------------------------------------------------------

	if plot == True:

		#- Time domain.

		plt.plot(t,h,'k')
		plt.xlim(0.0,float(nt)*dt)
		plt.xlabel('time [s]')
		plt.title('source time function (time domain)')

		plt.show()

		#- Frequency domain.

		hf = np.fft.fft(h)
		f = np.fft.fftfreq(len(hf), dt)

		plt.semilogx(f,np.abs(hf),'k')
		plt.plot([fmin,fmin],[0.0, np.max(np.abs(hf))],'r--')
		plt.text(1.1*fmin, 0.5*np.max(np.abs(hf)), 'fmin')
		plt.plot([fmax,fmax],[0.0, np.max(np.abs(hf))],'r--')
		plt.text(1.1*fmax, 0.5*np.max(np.abs(hf)), 'fmax')
		plt.xlim(0.1*fmin,10.0*fmax)
		plt.xlabel('frequency [Hz]')
		plt.title('source time function (frequency domain)')

		plt.show()

	#- Write to file. -----------------------------------------------------------------------------

	f = open(filename, 'w')

	#- Header.

	f.write('source time function, ses3d version 4.1\n')
	f.write('nt= '+str(nt)+', dt='+str(dt)+'\n')
	f.write('filtered Heaviside, highpass(fmin='+str(fmin)+', corners=3, zerophase=False), lowpass(fmax='+str(fmax)+', corners=5, zerophase=False)\n')
	f.write('-- samples --\n')

	for k in range(len(h)):
		f.write(str(h[k])+'\n')

	f.close()

        if get_stf == True: 
           return h
