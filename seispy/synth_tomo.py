#collection of functions for dealing with hypothetical tomographic inversions
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, lfilter, freqz

#class receiver_grid(object):
#   def __init__(self,type,Nx,Ny):

def ff_input(hypothetical_structure,phase_list,receiver_grid,frequency_list,N_rotations):
   '''
   args:
   hypothetical_structure: name of plume (or other heterogeneity) model
   phase_list: list of phases used in the inversion (currently just P or S)
   receiver_grid: hypothetical receiver array (should be instance of receiver_grid class)
   frequency_list: which passbands to use
   '''
   print 'fuck'




def filter_freqs(lowcut, highcut, fs, plot=False, corners=4):
   '''
   The frequency band information should technically be the amplitude spectrum
   of the filtered seismograms, but the amplitude spectrum of the bandpass
   filter is sufficient. Here we use a bandpass butterworth filter.

   The function freq band takes 3 arguments:
           lowcut = low end of bandpass (in Hz)
           highcut = high end of bandpass (in Hz)
           fs      = sample rate (1.0/dt)

   It returns two vectors:
           omega = frequency axis (in rad/s)
           amp   = frequency response of the filter
   '''
   #Define bandpass parameters
   nyquist = 0.5 * fs
   fmin    = lowcut/nyquist
   fmax    = highcut/nyquist

   #Make filter shape
   b, a = iirfilter(corners, [fmin,fmax], btype='band', ftype='butter')

   #Determine freqency response
   freq_range = np.linspace(0,0.15,200)
   w, h = freqz(b,a,worN=freq_range)

   omega    = fs * w                # in rad/s
   omega_hz = (fs * w) / (2*np.pi)  # in hz
   amp   = abs(h)

   if(plot == True):
      #Checks----------------
      #plt.semilogx(omega,amp)
      #plt.axvline(1/10.0)
      #plt.axvline(1/25.0)
      #plt.axhline(np.sqrt(0.5))

      plt.plot(omega,amp)
      plt.xlabel('frequency (rad/s)')
      plt.ylabel('amplitude')
      plt.show()

   return  omega, amp
