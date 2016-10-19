import os
import glob
import h5py
import obspy
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from seis_tools.seispy import filter
from seis_tools.seispy.data import cross_cor
from scipy import interpolate
from seis_tools.seispy.synth_tomo import filter_freqs

def delays(background_model,plume_model,phase,freqmin,freqmax,h5py_file,delta,**kwargs):
   '''
   Calculates delay times for ses3d synthetics and writes the output to an h5py file

   args--------------------------------------------------------------------------
   background_model:  specify path to waveform data of background model
   plume_model: specify path to waveform data of plume model
   phase: P or S delays (future will add other options such as SS and SKS)
   freqmin: minimum frequency of bandpass filter
   freqmax: maximum frequency of bandpass filter
   h5py_file: name of output file
   delay: distance in degrees between the earthquake and plume axis
   
   kwargs------------------------------------------------------------------------
   window: choose time window surrounding main phase.  By default the window will
           be [phase - T_max - 5, phase + T_max + 5], where T_max is the maximum period of
           the filter.
   distance_range: remove stations outside this range of epicentral distances
   filter_type: type of filter used (only bandpass at the moment)
   plot: show scatter plot of delay times. default=False
   taup_model: an instance of the obspy TauPyModel class, used to estimate phase
               arrivals.
   '''
   #get kwargs-------------------------------------------------------------------
   T_min = 1.0/freqmax
   T_max = 1.0/freqmin
   window = kwargs.get('window',(-1.0*T_max+5.0,T_max-5.0))
   distance_range = kwargs.get('distance_range',(20,90))
   filter_type = kwargs.get('filter_type','bandpass')
   plot = kwargs.get('plot',False)
   taup_model = kwargs.get('taup_model','default')
   print 'window = ', window, 'distance_range = ', distance_range, 'plot = ', plot

   #open h5py file---------------------------------------------------------------
   if os.path.isfile(h5py_file):
      output = h5py.File(h5py_file,'r+')
   else:
      output = h5py.File(h5py_file,'w')

   #read data--------------------------------------------------------------------
   #  the component to read is based on which phase is being cross correlated
   if phase == 'P':
      try:
         st1 = obspy.read(background_model+'st_Z.pk')
         st2 = obspy.read(plume_model+'st_Z.pk')
      except(IOError):
         st1 = obspy.read(background_model+'stz.pk')
         st2 = obspy.read(plume_model+'stz.pk')
   elif phase == 'S':
      try:
         st1 = obspy.read(background_model+'st_N.pk')
         st2 = obspy.read(plume_model+'st_N.pk')
      except(IOError):
         st1 = obspy.read(background_model+'stn.pk')
         st2 = obspy.read(plume_model+'stn.pk')
   elif phase == 'SKS':
      try:
         st1 = obspy.read(background_model+'st_E.pk')
         st2 = obspy.read(plume_model+'st_E.pk')
      except(IOError):
         st1 = obspy.read(background_model+'ste.pk')
         st2 = obspy.read(plume_model+'ste.pk')

   print st1,st2

   st1 = filter.range_filter(st1,distance_range)
   st2 = filter.range_filter(st2,distance_range)
   st1.sort()
   st2.sort()

   print 'streams after range filter ',st1,st2
   
   sampling_rate = st1[0].stats.sampling_rate
   st1.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=2)
   st2.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=2)
   #frequencies = filter_freqs(freqmin,freqmax,sampling_rate,plot=True,corners=2)
   
   #create lists for scattered data----------------------------------------------
   x = []
   y = []
   delays = []

   if phase == 'P':
      phase_list = ['P','Pdiff','p']
   elif phase == 'S':
      phase_list = ['S','Sdiff','s']
   elif phase == 'SKS':
      phase_list = ['SKS']
 
   print st1,st2
   for j in range(0,len(st1)):
      delay = cross_cor(st1[j],st2[j],phase=phase_list,window=window,taup_model=taup_model)
      print delay
      x.append(st1[j].stats.sac['stlo'])
      y.append(st1[j].stats.sac['stla'])
      delays.append(delay)

   lons = np.array(x)
   lats = np.array(y)
   delays = np.array(delays)
   data = np.array((lons,lats,delays))

   #plot maps-------------------------------------------------------------------- 
   if plot:
      plt.scatter(lons,lats,c=delays)
      plt.colorbar()
      plt.show()

   #write delays-----------------------------------------------------------------
   set_name = '{}/{}/{:<4.1f}'.format(phase,delta,T_min)
   dataset = output.create_dataset(set_name,data=data)
   dataset.attrs['filter'] = filter_type
   dataset.attrs['freqmin'] = freqmin
   dataset.attrs['freqmax'] = freqmax
   dataset.attrs['window'] = window
   output.close()
