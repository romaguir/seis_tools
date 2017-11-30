import scipy
import obspy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from obspy.taup import TauPyModel
model = TauPyModel(model="ak135")

###############################################################################
def phase_window(tr,phases,window_tuple,taup_model=model):
###############################################################################
    '''
    return window around phase
    '''
    tr.stats.distance = tr.stats.sac['gcarc']
    origin_time = tr.stats.sac['o']
    start = tr.stats.starttime
    time = taup_model.get_travel_times(source_depth_in_km = tr.stats.sac['evdp'],
                                       distance_in_degree = tr.stats.sac['gcarc'],
                                       phase_list = phases)
    debug = False

    if debug:
       print 'phases asked for : ', phases, 'at distance ',tr.stats.distance
       print 'arrivals : ',time

    t = origin_time + time[0].time

    windowed_tr = tr.slice(start+t+window_tuple[0],start+t+window_tuple[1])
    return windowed_tr

###############################################################################
def align_on_phase(st, **kwargs):
###############################################################################
    '''
    Use to precisely align seismogram on phase
    '''
    phase = kwargs.get('phase',['P'])
    def roll_zero(array,n):
        if n < 0:
            array = np.roll(array,n)
            array[n::] = 0
        else:
            array = np.roll(array,n)
            array[0:n] = 0
        return array

    for tr in st:
        arrivals = model.get_travel_times(distance_in_degree=tr.stats.sac['gcarc'],
                         source_depth_in_km=tr.stats.sac['evdp'],
                         phase_list = phase)
        P = arrivals[0]
        t = tr.stats.starttime
        o = tr.stats.sac['o']
        t+P.time+o
        window_data = (tr.slice(t+P.time-15+o,t+P.time+15+o).data)
        max_P = window_data[window_data < 0].min()
        imax = np.argmin(np.abs(max_P-window_data))
        shift = int(len(window_data)/2.)-imax
        tr.data = np.roll(tr.data,(1*shift))
    return st

###############################################################################
def normalize_on_phase(st,**kwargs):
###############################################################################
    '''
    normalize traces in stream based on maximum value in phase window
    '''
    phase = kwargs.get('phase',['P'])
    window_tuple = kwargs.get('window_tuple',(-10,10))

    for tr in st:
        window = phase_window(tr,phase,window_tuple)
        tr.data = tr.data/np.abs(window.data).max()
    return st


###############################################################################
def periodic_corr(data, deconvolution):
###############################################################################
    '''
    Periodic correlation, implemented using the FFT.
    data and deconvolution must be real sequences with the same length.

    Designed to align deconvolved trace with convolved trace.

    Use np.roll to shift deconvolution by value returned.
    '''
    corr = np.fft.ifft(np.fft.fft(data) * np.fft.fft(deconvolution).conj()).real
    shift = np.where(corr == corr.max())[0][0]
    return shift

###############################################################################
def waterlevel_deconvolve(tr):
###############################################################################
    '''
    Water_level_deconvolve trace
    Assume seismic wavelet is the P phase.
    works for both .xh and .sac formats
    '''

    def isolate_source(tr):
        '''
        Find source time function by multiplying P arrival with Tukey window
        '''

        stats_dict = {}

        if tr.stats._format == 'XH':
            stats_dict['depth'] = tr.stats.xh['source_depth_in_km']
            stats_dict['distance'] = float(tr.stats.station.split('_')[0])+float(tr.stats.station.split('_')[1])/100.
            stats_dict['start_time'] = tr.stats.starttime
            stats_dict['end_time'] = tr.stats.endtime
            stats_dict['offset'] = 0.
            stats_dict['sampling_rate'] = tr.stats.sampling_rate
        elif tr.stats._format == 'SAC':
            stats_dict['depth'] = tr.stats.sac['evdp']
            stats_dict['distance'] = tr.stats.sac['gcarc']
            stats_dict['start_time'] = tr.stats.starttime
            stats_dict['end_time'] = tr.stats.endtime
            stats_dict['offset'] = tr.stats.sac['o']
            stats_dict['sampling_rate'] = tr.stats.sampling_rate

        if stats_dict['end_time']-stats_dict['start_time'] <= 0:
            #return 'REMOVE'
            raise ValueError('starttime is larger than endtime')

        taup_time = model.get_travel_times(source_depth_in_km = stats_dict['depth'],
                                      distance_in_degree = stats_dict['distance'],
                                      phase_list = ['P'])
        P_arrival_time = taup_time[0].time+stats_dict['offset']


        begin_pad = tr.slice(stats_dict['start_time'],stats_dict['start_time']+P_arrival_time).data.size
        P_array = tr.slice(stats_dict['start_time']+P_arrival_time,stats_dict['start_time']+P_arrival_time+5).data.size
        end_pad = tr.slice(stats_dict['start_time']+P_arrival_time+5,stats_dict['end_time']).data.size

        tukey = scipy.signal.tukey(P_array,alpha=0.6)
        tukey_pad = np.pad(tukey,(begin_pad-1,end_pad-1),'constant',constant_values=(0,0))
        wavelet = tukey_pad*tr.data
        wavelet = np.roll(wavelet,-1*begin_pad-(P_array/2))

        return wavelet,tr.data,stats_dict

    def apply_waterlevel(tukey_pad,trace_data,alpha):
        '''
        Apply water level to fill spectral holes. see Practical Seismic Data Analysis
        page 182
        '''

        tukey_omega = np.fft.fft(tukey_pad)
        trace_omega = np.fft.fft(trace_data)

        F_omega = tukey_omega*tukey_omega.conjugate()
        F_omega[F_omega < alpha*F_omega.max()] = alpha*F_omega.max()
        out = (trace_omega*tukey_omega.conjugate())/F_omega
        out = np.fft.ifft(out)
        return out.real

    pad, data, stats_dict = isolate_source(tr)
    out = apply_waterlevel(pad,data,0.1)
    out = out-out.mean()
    out = out/out.max()

    if np.isnan(np.sum(out)):
        #return 'REMOVE'
        raise ValueError('NaN in deconvolved trace')

    return [stats_dict,out]

def cross_cor(tr1,tr2,mode='cross_cor',**kwargs):
   '''
   Cross correlation between two traces windowed around an expected arrival for a given
   phase.  The default reference model used to predict phase arrivals is ak135, but a 
   user defined model can be used instead.

   arguments:---------------------------------------------------------------------------
   tr1:         Trace 1 (obspy trace object)
   tr2:         Trace 2 (obspy trace object)
   mode:        cross correlation delay times: 'cross_cor' (default) or onset delay times: 'onset'

   **kwargs:----------------------------------------------------------------------------
   taup_model:  The velocity model used to predict arrivals. If 'default', ak135 if used.
                Otherwise, pass an instance of an obspy TauPyModel.
   filter:      Whether or not to filter the traces.
   fmin:        Minimum frequency used in bandpass, in Hz (if filter=True)
   fmax:        Maximum frequency used in bandpass, in Hz (if filter=True)
   window:      Time window around specified phase, in s.
   plot:        Whether or not to plot the traces in the cross correlation window. 
   onset_threshold: for use with 'onset' mode.  value onsidered to be the 'onset'
                    of the seismic phase, as a fraction of the maximum value of the
                    phase (i.e., 0.05 means we consider the onset to be where
                    the wave reaches 5% of the maximum amplitude of the arrival)

   Returns:-----------------------------------------------------------------------------
   delay:       Delay time, in s.
   '''
   from scipy.signal import correlate
   from seis_tools.seispy.data import phase_window

   #get kwargs
   filter = kwargs.get('filter',False)
   fmin   = kwargs.get('fmin',1.0/20.0)
   fmax   = kwargs.get('fmax',1.0/10.0)
   plot   = kwargs.get('plot',False)
   window = kwargs.get('window',(-20,20))
   phase  = kwargs.get('phase','P')
   taup_model = kwargs.get('taup_model','default')
   onset_threshold = kwargs.get('onset_threshold',0.05)

   #copy traces so you don't destroy them
   tr_1 = tr1.copy()
   tr_2 = tr2.copy()

   #get taup_model
   if taup_model == 'default':
      taup_model = model

   #filter
   if filter == True:
      tr_1.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)
      tr_2.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)

   #window the traces
   tr_1 = phase_window(tr_1,phases=phase,window_tuple=window,taup_model=taup_model)
   tr_2 = phase_window(tr_2,phases=phase,window_tuple=window,taup_model=taup_model)

   #calculate delay time
   if mode=='cross_cor':
      correlogram = correlate(tr_1.data,tr_2.data)
      delay = (np.argmax(correlogram)-len(correlogram)/2)*(1.0/tr_1.stats.sampling_rate)

   elif mode=='onset':
      tr_1.normalize()
      tr_2.normalize()
      arr1_i = 'nan'
      arr2_i = 'nan'

      for i in range(0,len(tr_1.data)):
          if np.abs(tr_1.data[i]) >= onset_threshold:
              arr1_i = i
              break
      for i in range(0,len(tr_2.data)):
          if np.abs(tr_2.data[i]) >= onset_threshold:
              arr2_i = i
              break

      shift_i = arr1_i - arr2_i
      delay = shift_i * (1.0/tr_1.stats.sampling_rate)

   #plot
   if plot == True:
      fig,axes = plt.subplots(2,figsize=(8,12))
      time = np.linspace(window[0],window[1],tr_1.stats.npts)
      axes[0].plot(time,tr_1.data)
      axes[0].plot(time,tr_2.data)
      axes[0].set_xlabel('time (s)')
      axes[0].set_title('delay = '+str(delay)+' s')
      #axes[1].plot(correlogram)
      #axes[1].set_xlabel('sample number') 
      #axes[1].set_title('correlelogram')
      plt.show()

   return delay

#################################################################################
def bootstrap_signals(signal_list,M,confidence_level=0.9):
#################################################################################
   '''
   params:
   signal_list: a list of the input signals to be stacked
   M: number of bootstrap realizations
   confidence_level: statistical confidence interval (based on normal distribution)

   returns:
   mean, std, conf_int
   '''
   N = len(signal_list)
   bootstrap_realization = []
   for i in range(0,M):
      rand = np.random.random_integers(0,N-1,N)
      stack = 0
      for j in rand:
         stack += signal_list[j]
      stack /= N
      bootstrap_realization.append(stack)

   A = np.vstack(bootstrap_realization)
   mean = A.mean(0)
   std = A.std(0)
   conf_int = stats.norm.interval(0.90,loc=mean,scale=std)
   
   return mean,std,conf_int

#################################################################################
def first_arrival(trace,phase,taup_model=model,threshold=0.05,return_delay=True,window=[-20,20]):
#################################################################################
   '''
   This is a quick and dirty way of estimating the first arrival
   on a trace.  The trace is first be windowed around the theoretical
   arrival, and normalized on the maximum of the
   phase of interest to make the threshold value meaningful

   trace: obspy trace object with event characteristics in sac dictionary
   phase: the phase of interest (should only use P or S)
   taup_model: an instance of the TauPyModel class
   threshold: value that is considered the onset of the arrival 
   return_delay: If True, returns the travel time delay.  If False, returns
                  the absolute travel time.
   window: time around to window phase (default 20 seconds before to 20 seconds
                                        after the theoretical arrival)
   '''
   from seis_tools.seispy.data import phase_window

   #avoid null arrivals
   if phase == 'P':
      phase_list = ['P','p','Pdiff']
   elif phase == 'S':
      phase_list = ['S','s','Sdiff']

   #get theoretical arrival
   arrs = taup_model.get_travel_times(source_depth_in_km=trace.stats.sac['evdp'],
                                      distance_in_degree=trace.stats.sac['gcarc'],
                                      phase_list = phase_list)
   predicted_time = arrs[0].time

   #window the trace
   trace = phase_window(trace,phases=phase_list,window_tuple=window)

   max_val = np.max(trace.data)
   min_val = np.min(trace.data)

   #normalize trace
   if np.abs(min_val) > max_val:
      trace.data /= min_val
   else:
      trace.data /= max_va

   arr_index = 0
   #get actual travel time
   for i in range(0,len(trace.data)):
      if np.abs(trace.data[i]) >= threshold:
         arr_index = i
         print "Found the value ", threshold, "at index", i
         break
   
   observed_time = predicted_time + trace.stats.delta*arr_index + window[0]
   print "predicted time", predicted_time
   print "observed time", observed_time

   if return_delay:
      return predicted_time - observed_time
   else:
      return observed_time

def get_gcarc(tr):
    '''
    takes a trace which has a sac dictionary and adds gcarc
    '''
