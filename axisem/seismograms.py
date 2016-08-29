import numpy as np
import matplotlib.pylab as plt
from scipy.signal import correlate

######################################################################
#                         seismogram class
######################################################################
class axisem_seismogram:

      def ___init___(self):
        
          self.nt = 100                            # number of samples
          self.dt = 0.1                            # delta t
          self.lat = 0.0                           # latitude of rec 
          self.lon = 0.0                           # longitude of rec
          self.t = 0.0*np.arange(10,dtype=float)   # time vector
          self.trace_x=0.0*np.arange(10,dtype=float)
          self.trace_y=0.0*np.arange(10,dtype=float)
          self.trace_z=0.0*np.arange(10,dtype=float)
          self.name = 'null'                       # station name
      
      ################################################################
      # read
      ################################################################

      def read(self, station_name, plot=True):

          seismogram = np.loadtxt(station_name) 
          self.t = seismogram[:,0]
          self.trace_x = seismogram[:,1]
          self.nt = len(self.t)
          self.dt = np.max(self.t)/len(self.t)

          #if (plot == true):
          
          plt.plot(self.t, self.trace_x)

 ################################################################
 #              cross correlate travel times
 ################################################################
          
def x_corr(a,b,center_time_s=1000.0,window_len_s=50.0,plot=True):

      center_index = int(center_time_s/a.dt)
      window_index = int(window_len_s/(a.dt))
      print "center_index is", center_index
      print "window_index is", window_index
     
      t1 = a.trace_x[(center_index - window_index) : (center_index + window_index)]
      t2 = b.trace_x[(center_index - window_index) : (center_index + window_index)]
      print t1

      time_window = np.linspace((-window_len_s/2.0), (window_len_s/2), len(t1))
      #print time_window
     
      #plt.plot(time_window, t1)
      #plt.plot(time_window, t2)
      #plt.show()

      x_corr_time = correlate(t1, t2)
      delay = (np.argmax(x_corr_time) - (len(x_corr_time)/2) ) * a.dt
      #print "the delay is ", delay
      return delay

