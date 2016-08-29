import obspy
import scipy
import numpy as np
import matplotlib.pyplot as plt
from data import phase_window
from data import align_on_phase
from obspy.taup import TauPyModel
from obspy.signal import rotate
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
from seis_tools.decon.water_level import water_level
from seis_tools.decon.damped_lstsq import damped_lstsq
#from seis_tools.seispy.decon import damped_lstsq
from obspy import UTCDateTime
import geopy
from geopy.distance import VincentyDistance

#testing deconvolution
#from rf.deconvolve import deconvt
#from seis_tools.seispy.decon import use_lsrn
#from seis_tools.seispy.decon import rf_time_decon
#from seis_tools.seispy.decon import solve_toeplitz

try:
    from obspy.geodetics import gps2dist_azimuth
except ImportError:    
    from obspy.core.util.geodetics import gps2DistAzimuth as gps2dist_azimuth
try:
   from obspy.geodetics import kilometer2degrees
except ImportError:
   from obspy.core.util.geodetics import kilometer2degrees

print "Using obspy version ",obspy.__version__
#-------------------------------------------------------------------------------------

#######################################################################################
#DEPENDENCIES:
# obspy
# obspyh5
#######################################################################################

#######################################################################################
class receiver_function(object):
#######################################################################################
   '''
   The receiver funciton class contains an obspy stream with the three components 
   of the receiver function.  The three trace objects in the stream are, in order:

   rf_st[0] : transverse component receiver function
   rf_st[1] : radial component receiver function
   rf_st[2] : z component receiver function (not usually used)
   '''

   ####################################################################################
   def __init__(self,tr_e,tr_n,tr_z,**kwargs):
   ####################################################################################
      '''
      Initialize receiver function

      params:
      tr_e  :obspy trace object, BHE channel
      tr_n  :obspy trace object, BHN channel
      tr_z  :obspy trace object, BHZ channel

      **kwargs:
      taup_model: TauPyModel instance.  Passing a pre-existing model speeds things up 
                  since it isn't necessary to initialize the the model when creating
                  the receiver function object.

      taup_model_name: If taup_model = 'none', you can tell it which model it use. The
                       default is prem_5km.

      window:  A tuple describing the time window of the receiver function (times given
               relative to P).
      '''
      #inherit taup model to avoid initializing the model for every receiver function
      taup_model = kwargs.get('taup_model','none')
      taup_model_name = kwargs.get('taup_model_name','ak135') #prem_5km if doing migrations

      if taup_model == 'none':
         self.model = TauPyModel(model=taup_model_name)
      else:
         self.model = taup_model

      #cut window centered on P phase
      self.window  = kwargs.get('window',[-10,150])
      self.tr_e    = phase_window(tr_e,phases=['P'],window_tuple=self.window,taup_model=self.model)
      self.tr_n    = phase_window(tr_n,phases=['P'],window_tuple=self.window,taup_model=self.model)
      self.tr_z    = phase_window(tr_z,phases=['P'],window_tuple=self.window,taup_model=self.model)
      self.time    = np.linspace(self.window[0],self.window[1],len(self.tr_e.data))
      self.dt      = 1.0/self.tr_e.stats.sampling_rate

      #make start time zero NOT SURE IF THIS WORKS!!! RM/ 4/13/16
      self.tr_e.starttime = 0
      self.tr_n.starttime = 0
      self.tr_z.starttime = 0

      #initialize obspy stream
      self.rf_st   = obspy.Stream(self.tr_e)
      self.rf_st  += self.tr_n
      self.rf_st  += self.tr_z

      self.gcarc   = self.tr_e.stats.sac['gcarc']
      self.evdp    = self.tr_e.stats.sac['evdp']
      self.pierce  = []

      #read slowness table for moveout correction
      self.slowness_table = np.loadtxt('/geo/work10/romaguir/seismology/seis_tools/seispy/slowness_table.dat')

      #get slowness and predicted P410s, P660s arrival times
      tt = self.model.get_travel_times(source_depth_in_km = self.evdp,
                                       distance_in_degree = self.gcarc,
                                       phase_list=['P','P410s','P660s'])

      #just in case there's more than one phase arrival, loop through tt list
      for i in range(0,len(tt)):
         if tt[i].name == 'P':
            self.predicted_p_arr = tt[i].time 
            #ray parameter (horizontal slowness) of incident plane wave 
            self.ray_param = tt[i].ray_param_sec_degree
         elif tt[i].name == 'P410s':
            self.predicted_p410s_arr = tt[i].time 
         if tt[i].name == 'P660s':
            self.predicted_p660s_arr = tt[i].time 

      #event information
      self.evla  = self.tr_e.stats.sac['evla']
      self.evlo  = self.tr_e.stats.sac['evlo']
      self.stla  = self.tr_e.stats.sac['stla']
      self.stlo  = self.tr_e.stats.sac['stlo']
      self.gcarc = self.tr_e.stats.sac['gcarc']
      self.evdp  = self.tr_e.stats.sac['evdp']

   ####################################################################################
   def plot(self):
   ####################################################################################
      #make axes-----------------------------------------------------------------------
      fig,axes = plt.subplots(3,sharex=True,figsize=(12,6)) 
      font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
           }

      #plot data-----------------------------------------------------------------------
      axes[0].plot(self.time,self.rf_st[0].data,'k') 
      axes[0].grid()
      axes[0].ticklabel_format(style='sci',scilimits=(0,1),axis='y')
      axes[1].plot(self.time,self.rf_st[1].data,'k') 
      axes[1].grid()
      axes[1].ticklabel_format(style='sci',scilimits=(0,1),axis='y')
      axes[2].plot(self.time,self.rf_st[2].data,'k') 
      axes[2].grid()
      axes[2].ticklabel_format(style='sci',scilimits=(0,1),axis='y')

      #plot expected P410s and P660s arrivals
      t410 = self.predicted_p410s_arr - self.predicted_p_arr
      t660 = self.predicted_p660s_arr - self.predicted_p_arr
      axes[0].axvline(t410, color='r')
      axes[0].axvline(t660, color='r')
      axes[1].axvline(t410, color='r')
      axes[1].axvline(t660, color='r')
      axes[2].axvline(t410, color='r')
      axes[2].axvline(t660, color='r')

      #labels--------------------------------------------------------------------------
      axes[0].set_ylabel('amplitude',fontdict=font)
      axes[0].text(0.05,0.85,self.rf_st[0].stats.channel,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[0].transAxes,
                 fontdict=font)
      axes[1].set_ylabel('amplitude',fontdict=font)
      axes[1].text(0.05,0.85,self.rf_st[1].stats.channel,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[1].transAxes,
                 fontdict=font)
      axes[2].set_ylabel('amplitude',fontdict=font)
      axes[2].set_xlabel('time after P (s)',fontdict=font)
      axes[2].text(0.05,0.85,self.rf_st[2].stats.channel,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[2].transAxes,
                 fontdict=font)
      plt.show()

   ####################################################################################
   def rotate(self,rotation_method='RTZ'):
   ####################################################################################

       #rotate-------------------------------------------------------------------------
       for i in range(0,len(self.rf_st)):
          self.rf_st[i].stats.back_azimuth = self.tr_e.stats.sac['baz']

       self.rf_st.rotate(method='NE->RT')
  
       if rotation_method == 'LQT':
          r_amp           = np.amax(np.amax(self.rf_st[1].data))
          z_amp           = np.amax(np.amax(self.rf_st[2].data))
          incidence_angle = np.arctan(r_amp/z_amp) * (180.0/np.pi)
          
          for i in range(0,len(self.rf_st)):
             self.rf_st[i].stats.inclination = incidence_angle

          self.rf_st.rotate(method='RT->NE')
          self.rf_st.rotate(method='ZNE->LQT')

   ####################################################################################
   def get_prf(self,decon_type='water_level',wl=0.1,damping=5.0,rotation_method='RTZ'):
   ####################################################################################

       #rotate-------------------------------------------------------------------------
       for tr in self.rf_st:
          tr.stats.back_azimuth = tr.stats.sac['baz']

       for tr in self.rf_st:
          tr.stats.starttime = 0

       self.rf_st.rotate(method='NE->RT')
  
       if rotation_method == 'LQT':
          r_amp           = np.amax(np.amax(self.rf_st[1].data))
          z_amp           = np.amax(np.amax(self.rf_st[2].data))
          incidence_angle = np.arctan(r_amp/z_amp) * (180.0/np.pi)

          print 'using LQT rotation'
          
          for i in range(0,len(self.rf_st)):
             self.rf_st[i].stats.inclination = incidence_angle

          self.rf_st.rotate(method='RT->NE')
          self.rf_st.rotate(method='ZNE->LQT')

       #deconvolve---------------------------------------------------------------------
       #divisor should be the L or Z component (depending on rotation method) 
       div = self.rf_st[2].data
       if decon_type == 'water_level':
          #self.rf_st[0].data = water_level(self.rf_st[0].data,div,alpha=wl)
          self.rf_st[1].data = water_level(self.rf_st[1].data,div,alpha=wl)
          #self.rf_st[2].data = water_level(self.rf_st[2].data,div,alpha=wl)
       elif decon_type == 'damped_lstsq':
          #self.rf_st[0].data = damped_lstsq(self.rf_st[0].data,div,damping=damp)

          #self.rf_st[1].data = damped_lstsq(self.rf_st[1].data,div,damping=damping)
          self.rf_st[1].data = damped_lstsq(div,self.rf_st[1].data,damping=damping)

          #self.rf_st[2].data = damped_lstsq(self.rf_st[2].data,div,damping=damp)

       #TESTING FASTER DECON METHODS 5/24/16
       #elif decon_type == 'rf_time_decon':
       #   self.rf_st[1].data = rf_time_decon(self.rf_st[1].data,div)
       #elif decon_type == 'solve_toeplitz':
       #   self.rf_st[1].data = solve_toeplitz(div,self.rf_st[1].data)
       elif decon_type == 'use_lsrn':
          self.rf_st[1].data = use_lsrn(self.rf_st[1].data,div)
          

       #center on P--------------------------------------------------------------------
       spike       = np.exp((-1.0*(self.time)**2)/0.1)
       spike_omega = np.fft.fft(spike)

       for i in range(0,len(self.rf_st)):
          data_omega         = np.fft.fft(self.rf_st[i].data)
          shifted            = spike_omega*data_omega
          self.rf_st[i].data = np.real(np.fft.ifft(shifted))

       #normalize on maximum of L component
       peak_amp = np.max(self.rf_st[2].data)
       for rf in self.rf_st:
         rf.data /= peak_amp

   #TODO write a function to reconvolve and compare misfit
   ####################################################################################
   def check_decon(self):
   ####################################################################################
      recon = np.convolve(self.rf_st[1],self.tr_z)
      plt.plot(self.time,recon)
      plt.plot(self.time,self.tr_e)

   ####################################################################################
   def moveout_correction(self):
   ####################################################################################
      '''
      Moveout correction relative to a reference ray parameter of 6.4 s/deg.  This stretches
      the time axis for smaller ray parameters (larger epicentral distances), and 
      shrinks the time axis for larger ray parameters (smaller epicentral distances).

      #NOTE 3-16-16, Moveout correction doesn't work properly... The time axis seems
      to be stretching in the opposite way that it should.
      '''
      p = self.slowness_table[:,0]
      s = self.slowness_table[:,1]
      
      #interpolate with np.interp. make sure values in the first vector are increasing
      scale = np.interp(self.ray_param,p[::-1],s[::-1])
      #print "ray parameter, scale = ",self.ray_param,scale

      #scale the receiver function and interpolate new time axis
      new_time = self.time * scale

      f        = interp1d(new_time,self.rf_st[0].data,bounds_error=False,fill_value=0)
      self.rf_st[0].data = f(self.time)

      f        = interp1d(new_time,self.rf_st[1].data,bounds_error=False,fill_value=0)
      self.rf_st[1].data = f(self.time)

      f        = interp1d(new_time,self.rf_st[2].data,bounds_error=False,fill_value=0)
      self.rf_st[2].data = f(self.time)

   ####################################################################################
   def migrate_1d(self,**kwargs):
   ####################################################################################
      depth   = kwargs.get('depth_range',np.arange(50,1005,5))
      amp     = np.zeros((len(depth)))
      origin  = geopy.Point(self.evla,self.evlo)
      bearing = self.rf_st[0].stats.sac['az']

      ii = 0
      for d in depth:
         phase  = 'P'+str(d)+'s'
         pierce = self.model.get_pierce_points(self.evdp,self.gcarc,phase_list=[phase])
         arrs   = self.model.get_travel_times(self.evdp,self.gcarc,phase_list=['P',phase])

         #in case there's duplicate phase arrivals
         for arr in arrs:
           if arr.name == 'P':
              p_arr = arr.time
           elif arr.name == phase:
              pds_arr = arr.time

         #determine amplitude at each depth
         window_start = self.window[0]
         pds_minus_p  = pds_arr - p_arr
         i_start      = int((0.0 - window_start)/self.dt)
         i_t          = int(pds_minus_p/self.dt) + i_start
         amp[ii]      = self.rf_st[1].data[i_t]

         #find pierce points and create pierce dictionary
         points = pierce[0].pierce
         for p in points:
            if p['depth'] == d and np.degrees(p['dist']) > 20.0:
               prc_dist = np.degrees(p['dist'])
               d_km     = prc_dist * ((2*np.pi*6371.0/360.0))
               destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
               lat = destination[0]
               lon = destination[1]
               row = {'depth':d,'dist':prc_dist,'lat':lat,'lon':lon,'amplitude':amp[ii]}
               self.pierce.append(row)
         ii += 1

   ####################################################################################
   def plot_pierce_points(self,depth=410.0,ax='None',**kwargs):
   ####################################################################################
      '''
      Plots pierce points for a given depth.  If an axes object is supplied
      as an argument it will use it for the plot.  Otherwise, a new axes
      object is created.

      kwargs:
             depth:         pierce point depth
             proj:          map projection.  if none given, a default basemap axis is
                            made, defined by the coordinates of the corners. if 'ortho'
                            is given, a globe centered on Lat_0 and Lon_0 is made.
             ll_corner_lat: latitude of lower left corner of map
             ll_corner_lon: longitude of lower left corner of map
             ur_corner_lat: latitude of upper right corner of map
             Lat_0:         latitude center of ortho
             Lon_0:         logitude center of ortho
             return_ax:     whether or you return the basemap axis object. default False

      '''
      proj=kwargs.get('proj','default')
      ll_lat=kwargs.get('ll_corner_lat',-35.0)
      ll_lon=kwargs.get('ll_corner_lon',0.0)
      ur_lat=kwargs.get('ur_corner_lat',35.0)
      ur_lon=kwargs.get('ur_corner_lon',120.0)
      Lat_0=kwargs.get('Lat_0',0.0)
      Lon_0=kwargs.get('Lon_0',0.0)
      return_ax =kwargs.get('return_ax',False)

      if ax == 'None':
         if proj == 'default':
            m = Basemap(llcrnrlon=ll_lon,llcrnrlat=ll_lat,urcrnrlon=ur_lon,urcrnrlat=ur_lat)
         elif proj == 'ortho':
            m = Basemap(projection='ortho',lat_0=Lat_0,lon_0=Lon_0)
         m.drawmapboundary()
         m.drawcoastlines()
         m.fillcontinents()
      else:
         m = ax

      found_points = False
      for ii in self.pierce:
         if ii['depth'] == depth:
            x,y = m(ii['lon'],ii['lat'])
            found_points = True

      if found_points == True:
         m.scatter(x,y,100,marker='+',color='r',zorder=99)
      else:
         print "no pierce points found for the given depth"


   ###############################################################################
   def shift(self,phase,ref_deg=64.0):
   ###############################################################################
      '''
      Shifts the time axis to account for moveout of a given phase
      
      params---------------------------------------------------------------------
      ref_deg: float, reference epicentral distance
      phase:   string, phase name for which to apply moveout correction
      '''
      t_ref = self.model.ge_travel_times(source_depth_in_km=self.evdp,
                                          distance_in_degree=ref_deg,
                                          phase_list=["P",phase])
      t_arr = self.model.get_travel_times(source_depth_in_km=self.evdp,
                                          distance_in_degree=self.gcarc,
                                          phase_list=["P",phase])
      for arr in t_ref:
         if arr.name=='P':
            p_arrival_ref = arr.time
         elif arr.name==phase:
            pds_arrival_ref = arr.time
        
      for arr in t_arr:
         if arr.name=='P':
            p_arrival = arr.time
         elif arr.name==phase:
            pds_arrival = arr.time

      time_shift = (pds_arrival_ref-p_arrival_ref)-(pds_arrival-p_arrival)
      int_shift  = int(time_shift/self.dt)
      self.rf_st[0].data = np.roll(self.rf_st[0].data,int_shift)
      self.rf_st[1].data = np.roll(self.rf_st[1].data,int_shift)
      self.rf_st[2].data = np.roll(self.rf_st[2].data,int_shift)

   ##############################################################################
   def zero_pP(self,**kwargs):
   ##############################################################################
      '''
      Finds predicted pP arrival and zeros a window centered on the arrival

      kwargs---------------------------------------------------------------------
      window_half_dur : half duration of zero window (default = 2.5 s)
      '''
      window_half_dur = kwargs.get('window_half_dur',2.5)
      arrs = self.model.get_travel_times(source_depth_in_km=self.evdp,
                                         distance_in_degree=self.gcarc,
                                         phase_list=['P','pP'])
      P_arr  = 'none'
      pP_arr = 'none'

      for arr in arrs:
         if arr.name == 'P':
            P_arr = arr
         elif arr.name == 'pP': 
            pP_arr = arr

      if P_arr == 'none' or pP_arr == 'none':
         raise ValueError('problem occured in function "zero_pP", no matching arrivals found')
      else:
         P_time = P_arr.time
         pP_time = pP_arr.time
         delay_time = pP_time - P_time
         zero_window_center = -1.0*self.window[0] + delay_time
         zero_window_start  = zero_window_center - window_half_dur
         zero_window_end    = zero_window_center + window_half_dur
         zero_window_start_index = int(zero_window_start/self.dt)
         zero_window_end_index   = int(zero_window_end/self.dt)

         #case 1: entire window is in range
         if zero_window_start_index >= 0 and zero_window_end_index <= len(self.rf_st[1].data):
            self.rf_st[1].data[zero_window_start_index:zero_window_end_index] = 0.0
         #case 2: end of window is out of range
         if zero_window_start_index >= 0 and zero_window_end_index >= len(self.rf_st[1].data):
            self.rf_st[1].data[zero_window_start_index:] = 0.0
         #case 3: entire window is out of range
         if zero_window_start_index >= len(self.rf_st[1].data):
            print "pP arrives outside the receiver function window"

   ##############################################################################
   def zero_PP(self,**kwargs):
   ##############################################################################
      '''
      Finds predicted PP arrival and zeros a window centered on the arrival

      kwargs---------------------------------------------------------------------
      window_half_dur : half duration of zero window (default = 2.5 s)
      '''
      window_half_dur = kwargs.get('window_half_dur',2.5)
      arrs = self.model.get_travel_times(source_depth_in_km=self.evdp,
                                         distance_in_degree=self.gcarc,
                                         phase_list=['P','PP'])
      P_arr  = 'none'
      PP_arr = 'none'

      for arr in arrs:
         if arr.name == 'P':
            P_arr = arr
         elif arr.name == 'PP':
            PP_arr = arr

      if P_arr == 'none' or PP_arr == 'none':
         raise ValueError('problem occured in function "zero_PP", no matching arrivals found')
      else:
         P_time = P_arr.time
         PP_time = PP_arr.time
         delay_time = PP_time - P_time
         zero_window_center = -1.0*self.window[0] + delay_time
         zero_window_start  = zero_window_center - window_half_dur
         zero_window_end    = zero_window_center + window_half_dur
         zero_window_start_index = int(zero_window_start/self.dt)
         zero_window_end_index   = int(zero_window_end/self.dt)

         #case 1: entire window is in range
         if zero_window_start_index >= 0 and zero_window_end_index <= len(self.rf_st[1].data):
            self.rf_st[1].data[zero_window_start_index:zero_window_end_index] = 0.0
         #case 2: end of window is out of range
         if zero_window_start_index >= 0 and zero_window_end_index >= len(self.rf_st[1].data):
            self.rf_st[1].data[zero_window_start_index:] = 0.0
         #case 3: entire window is out of range
         if zero_window_start_index >= len(self.rf_st[1].data):
            print "PP arrives outside the receiver function window"

   

##################################################################
#signal to noise test
##################################################################
def s2nr(trace,**kwargs):
   from numpy import mean, sqrt, square, arange
   '''
   Test to see if the signal to noise ratio satisfies the criteria
   as defined by Schmandt et al. 2012.  The criteria is that the
   RMS amplitudes of a 4 s window must by at least 3 times as large
   as a 16 s window.

   params-------------------------------------------------------
   trace: an obspy trace object

   kwargs-------------------------------------------------------
   windows: the length in seconds of the two windows which are
            compared.  dtype = tuple.  default: [4,16]
   time_before_P: how long the trace starts before the P arrival.
                  default: 10.0
   '''
   windows = kwargs.get('windows',[4,16])
   time_before_P = kwargs.get('time_before_P',10.0)

   trace_1 = trace.copy()
   trace_2 = trace.copy()
   t1_start = UTCDateTime(0.0)+time_before_P
   t2_start = UTCDateTime(0.0)+time_before_P+windows[0]
   t1_end = t1_start + windows[0]
   t2_end = t1_start + windows[1]

   signal1 = trace_1.trim(t1_start,t1_end)
   signal2 = trace_2.trim(t2_start,t2_end)

   rms1 = sqrt(mean(square(signal1.data)))
   rms2 = sqrt(mean(square(signal2.data)))
   rms_ratio = rms1/rms2

   return rms_ratio

##################################################################
def write_h5py_dict(trace,signal_to_noise_ratio,rf_id,ray_param):
##################################################################
   '''
   obspyh5 is a desirable way to store many traces in a single file, but the 
   sac dictionary is not preserved when writing to this format.  This function
   copies the contents of the sac dictionary so that they are preserved when 
   writing to obspyh5.

   args----------------------------------------------------------
   trace: an obspy trace (the reciever function)
   signal_to_noise_ratio
   rf_id: an identifier for the h5py dictionary (use an integer)
   ray_param: ray parameter in s/deg
   '''

   tr = trace.copy()
   tr.stats.evla      = tr.stats.sac['evla']
   tr.stats.evlo      = tr.stats.sac['evlo']
   tr.stats.evdp      = tr.stats.sac['evdp']
   tr.stats.stla      = tr.stats.sac['stla']
   tr.stats.stlo      = tr.stats.sac['stlo']
   tr.stats.gcarc     = tr.stats.sac['gcarc']
   tr.stats.baz       = tr.stats.sac['baz']
   tr.stats.az        = tr.stats.sac['az']
   tr.stats.o         = tr.stats.sac['o']
   tr.stats.ray_param = ray_param
   tr.stats.snr       = signal_to_noise_ratio
   tr.stats.rf_id     = rf_id

   return tr

##################################################################
#Tools for dealing with a list of receiver function objects
##################################################################
def delay_and_sum(rf_list,Phase):
   '''
   Takes a list of receiver function objects and stacks along the
   moveout of a given phase.

   params-------------------------------------------------------
   phase:   string, phase name for which to apply moveout correction
   '''
   if Phase != 'P':
      for rf in rf_list:
         rf.shift(phase=Phase)

   E_stack = np.zeros(len(rf_list[0].trace_e))
   N_stack = np.zeros(len(rf_list[0].trace_n))
   Z_stack = np.zeros(len(rf_list[0].trace_z))

   for rf in rf_list:
      E_stack += rf.trace_e
      N_stack += rf.trace_n
      Z_stack += rf.trace_z
      
   return N_stack

def vinnik77_beam_forming(rf_list):
   import copy
   '''
   function to do Vinnik 1977 style beamforming.
   currently in testing...

   the problem is that each call of "delay and sum"
   shifts the time axes of the receiver function list
   without shifting it back.
   '''

   rf0 = rf_list[0]
   #stack along different moveout curves


   #Straight stack:
   rfs = copy.deepcopy(rf_list)
   ss0 = delay_and_sum(rfs,Phase='P')

   #P200s stack:
   rfs = copy.deepcopy(rf_list)
   ss1 = delay_and_sum(rfs,Phase='P200s')

   #P400s stack:
   rfs = copy.deepcopy(rf_list)
   ss2 = delay_and_sum(rfs,Phase='P400s')

   #P600s stack:
   rfs = copy.deepcopy(rf_list)
   ss3 = delay_and_sum(rfs,Phase='P600s')

   #P800s stack:
   rfs = copy.deepcopy(rf_list)
   ss4 = delay_and_sum(rfs,Phase='P800s')

   #P1000s stack:
   rfs = copy.deepcopy(rf_list)
   ss5 = delay_and_sum(rfs,Phase='P1000s')

   #pP stack
   #rfs = copy.deepcopy(rf_list)
   #ss6 = delay_and_sum(rfs,Phase='pP')
  
   #stretching
   rfs = copy.deepcopy(rf_list)
   ss6 = np.zeros((len(ss0)))
   for rf in rfs:
      rf.moveout_correction()
      ss6 += rf.rf_st[1]

   #Plots
   max_amp = np.max(ss0)
   min_amp = np.min(ss0)
   t_start = rf0.time[0]
   t_end   = rf0.time[::-1][0]
   ax_lim  = [t_start,t_end,min_amp,max_amp]

   fig,axes = plt.subplots(7,sharex=True,sharey=True)
   axes[0].plot(rf0.time,ss0)
   axes[0].text(0.85,0.85,"P",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[0].transAxes)
   axes[0].axis(ax_lim)
   axes[1].plot(rf0.time,ss1)
   axes[1].text(0.85,0.85,"P200s",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[1].transAxes)
   axes[1].axis(ax_lim)
   axes[2].plot(rf0.time,ss2)
   axes[2].text(0.85,0.85,"P400s",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[2].transAxes)
   axes[2].axis(ax_lim)
   axes[3].plot(rf0.time,ss3)
   axes[3].text(0.85,0.85,"P600s",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[3].transAxes)
   axes[3].axis(ax_lim)
   axes[4].plot(rf0.time,ss4)
   axes[4].text(0.85,0.85,"P800s",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[4].transAxes)
   axes[4].axis(ax_lim)
   axes[5].plot(rf0.time,ss5)
   axes[5].text(0.85,0.85,"P1000s",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[5].transAxes)
   axes[5].axis(ax_lim)
   axes[6].plot(rf0.time,ss6)
   axes[6].text(0.85,0.85,"all",
              horizontalalignment='center',
              verticalalignment='center',
              transform=axes[6].transAxes)
   axes[6].axis(ax_lim)

   plt.show()

####################################################################################
def rf_moveout_correction(rf_trace,table='None'):
####################################################################################
   '''
   takes a receiver function trace in the rfh5 format. if table = 'None', the slowness
   lookup table will be read.  alternatively if calling this function repeatedly, pass 
   the table as an argument to avoid repetative i/o.
   '''

   if table == 'None':
      slowness_table = np.loadtxt('/geo/work10/romaguir/seismology/seis_tools/seispy/slowness_table.dat')
   else:
      slowness_table = table

   p = slowness_table[:,0]
   s = slowness_table[:,1]

   #interpolate with np.interp. make sure values in the first vector are increasing
   scale = np.interp(rf_trace.stats.ray_param,p[::-1],s[::-1])

   #scale the receiver function and interpolate new time axis
   time = np.linspace(0,rf_trace.stats.delta*rf_trace.stats.npts,rf_trace.stats.npts)
   new_time = time * scale

   f        = interp1d(new_time,rf_trace.data,bounds_error=False,fill_value=0)
   rf_mvc   = f(time)
   rf_trace.data = rf_mvc

   return rf_trace

####################################################################################
def migrate_1d(rf_trace,**kwargs):
####################################################################################
   '''
   takes an rf trace and returns a dictionary with pierce points and associated
   reciever function ampltiudes

   *note it's best to pass a TauPyModel instance (eg, prem_5km), to avoid having 
    to initiate a new model every time you call this function
   '''

   #get kwargs
   depth      = kwargs.get('depth_range',np.arange(50,805,5))
   taup_model = kwargs.get('taup_model','None')
   format     = kwargs.get('format','rfh5')
   window     = kwargs.get('window',[-10,100])

   #geographical information
   if format == 'rfh5':
      gcarc = rf_trace.stats.gcarc
      dt    = rf_trace.stats.delta
      evla  = rf_trace.stats.evla
      evlo  = rf_trace.stats.evlo
      evdp  = rf_trace.stats.evdp
      stla  = rf_trace.stats.stla
      stlo  = rf_trace.stats.stlo
      az    = rf_trace.stats.az
      o     = rf_trace.stats.o

   #initializations
   amp           = np.zeros((len(depth)))
   origin        = geopy.Point(evla,evlo)
   bearing       = az
   pierce_dict   = []
   if taup_model == 'None':
      taup_model = TauPyModel('prem_5km')

   ii = 0
   for d in depth:
      phase  = 'P'+str(d)+'s'
      pierce = taup_model.get_pierce_points(evdp,gcarc,phase_list=[phase])
      arrs   = taup_model.get_travel_times(evdp,gcarc,phase_list=['P',phase])

      #in case there's duplicate phase arrivals
      P_arrs = []
      Pds_arrs = []
      for arr in arrs:
        if arr.name == 'P':
           P_arrs.append(arr)
           #p_arr = arr.time
        elif arr.name == phase:
           #pds_arr = arr.time
           Pds_arrs.append(arr)
      p_arr = P_arrs[0].time 
      pds_arr = Pds_arrs[0].time

      #determine amplitude at each depth
      #window_start = o #TODO update writeh5py_dict so that win_start win_end are written
      pds_minus_p  = pds_arr - p_arr
      i_start      = int((0.0 - window[0])/dt)
      i_t          = int(pds_minus_p/dt) + i_start
      amp[ii]      = rf_trace.data[i_t]

      #find pierce points and create pierce dictionary
      points = pierce[0].pierce
      for p in points:
         if p['depth'] == d and np.degrees(p['dist']) > 20.0:
            prc_dist = np.degrees(p['dist'])
            d_km     = prc_dist * ((2*np.pi*6371.0/360.0))
            destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
            lat = destination[0]
            lon = destination[1]
            row = {'depth':d,'dist':prc_dist,'lat':lat,'lon':lon,'amplitude':amp[ii]}
            pierce_dict.append(row)
      ii += 1

   return pierce_dict

##################################################################################
def pds_time(evdp,gcarc,depth,model_1d):
##################################################################################
   '''
   Calculates the travel time of a Pds phase based on the spherical travel time 
   equation (e.g., Eager et al. 2010)

   args--------------------------------------------------------------------------
   evdp:event depth 
   gcarc:great circle distance
   depth:the conversion depth
   model_1d:velocity model 
   '''
   taup_model = TauPyModel(model_1d)
   p_arrs = taup_model.get_travel_times(evdp,gcarc,['P'])

##############################################################################
def zero_phase(tr,phase,rf_window=[-10,120],**kwargs):
##############################################################################
   '''
   Finds predicted PP arrival and zeros a window centered on the arrival
   args--------------------------------------------------------------------------
   tr: obspy trace
   phase: phase to zero
   time_start: starting point of receiver function time window

   kwargs---------------------------------------------------------------------
   window_half_dur : half duration of zero window (default = 2.5 s)
   taup_model
   '''
   window_half_dur = kwargs.get('window_half_dur',2.5)
   taup_model = kwargs.get('taup_model','none')
   if taup_model == 'none':
      taup_model = TauPyModel('prem')

   arrs = taup_model.get_travel_times(source_depth_in_km=tr.stats.evdp,
                                      distance_in_degree=tr.stats.gcarc,
                                      phase_list=['P',phase])
   print 'arrs = ', arrs
   P_arr  = 'none'
   phase_arr = 'none'

   for arr in arrs:
      if arr.name == 'P':
         P_arr = arr
      elif arr.name == phase:
         phase_arr = arr

   if P_arr == 'none' or phase_arr == 'none':
      raise ValueError('problem occured in function "zero_phase", no matching arrivals found')
   else:
      P_time = P_arr.time
      phase_time = phase_arr.time
      delay_time = phase_time - P_time
      zero_window_center = -1.0*rf_window[0] + delay_time
      zero_window_start  = zero_window_center - window_half_dur
      zero_window_end    = zero_window_center + window_half_dur
      zero_window_start_index = int(zero_window_start/tr.stats.delta)
      zero_window_end_index   = int(zero_window_end/tr.stats.delta)

      #case 1: entire window is in range
      if zero_window_start_index >= 0 and zero_window_end_index <= len(tr.data):
         tr.data[zero_window_start_index:zero_window_end_index] = 0.0
      #case 2: end of window is out of range
      if zero_window_start_index >= 0 and zero_window_end_index >= len(tr.data):
         tr.data[zero_window_start_index:] = 0.0
      #case 3: entire window is out of range
      if zero_window_start_index >= len(tr.data):
         print "PP arrives outside the receiver function window"

