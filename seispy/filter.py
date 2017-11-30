import scipy
import obspy
from obspy.taup import TauPyModel
import numpy as np
try:
    from obspy.geodetics import gps2dist_azimuth
except ImportError:
    from obspy.core.util.geodetics import gps2DistAzimuth as gps2dist_azimuth
try:
   from obspy.geodetics import kilometer2degrees
except ImportError:
   from obspy.core.util.geodetics import kilometer2degrees
#-------------------------------------------------------------------------------------


model = TauPyModel(model="ak135")

'''
Samuel Haugland 01/19/16

seis_filter.py includes functions needed to remove unwanted traces from
streams based on various criteria. All functions should take a stream object
and arguments and return the filtered stream object
'''

def kurtosis_filter(st, **kwargs):
    '''
    remove traces from phase based on kurtosis
    '''

    alpha = kwargs.get('alpha', False)
    if alpha is not False:
        alpha = 0.5

    k = []
    for tr in st:
        ki = scipy.stats.kurtosis(tr.data)
        if np.isnan(ki):
            st.remove(tr)
            continue
        else:
            k.append(ki)
    mean_k = sum(k)/len(st)

    for tr in st:
        if scipy.stats.kurtosis(tr.data) < (alpha*mean_k):
            st.remove(tr)
    return st


def dirty_filter(st,**kwargs):
    '''
    Remove trace from stream if noise is too much
    a,b  refer to the time windows before the phase
    c,d  refer to the time windows after the phase
    '''

    a = kwargs.get('a',50)
    b = kwargs.get('b',20)
    c = kwargs.get('c',10)
    d  = kwargs.get('d',30)

    phase = kwargs.get('phase',['P'])

    pre_limit = kwargs.get('pre_limit',0.3)
    post_limit = kwargs.get('post_limit',0.3)

    for tr in st:
        arrivals = model.get_travel_times(
                   distance_in_degree=tr.stats.sac['gcarc'],
                   source_depth_in_km=tr.stats.sac['evdp'],
                   phase_list=phase)

        P = arrivals[0]
        t = tr.stats.starttime
        o = tr.stats.sac['o']
        max_P = abs(tr.slice(t+P.time-10+o, t+P.time+10+o).data).max()
        pre_noise = abs(tr.slice(t+P.time-a+o, t+P.time-b+o).data).max()
        post_noise = abs(tr.slice(t+P.time+c+o, t+P.time+d+o).data).max()
        if (pre_noise > max_P*pre_limit) or (post_noise > max_P*post_limit):
            st.remove(tr)

    return st

def gimp_filter(st):
    '''
    Removes seismograms from trace if they have lengths too short. Makes all
    seismograms the same length and same sampling rate
    '''

    def max_len(st):
        a = []
        for tr in st:
            a.append(tr.data.shape[0])
        return max(a)

    def min_len(st):
        a = []
        for tr in st:
            a.append(tr.data.shape[0])
        return min(a)

    for tr in st:
        if tr.data.shape[0] < 100:
            st.remove(tr)

    #st.interpolate(sampling_rate=20.0)
    st.resample(sampling_rate=40.0)

    mx_len = max_len(st)

    for tr in st:
        if tr.data.shape[0] < mx_len-10:
            st.remove(tr)

    mn_len = min_len(st)

    for tr in st:
        tr.data = tr.data[0: mn_len]

    return st

def range_filter(st, range_tuple):
    '''
    Removes seismograms from trace if they fall outside the range limits
    of range_tuple

    range_tuple = (30,50) removes all traces outside of 30 to 50 degrees from
    source
    '''

    for tr in st:
        if not range_tuple[0] <= tr.stats.sac['gcarc'] <= range_tuple[1]:
            st.remove(tr)

    return st

def az_filter(st, az_tuple):
    '''
    Removes seismograms from stream if they fall outside the azimuth limits
    of az_tuple
    '''

    for tr in st:
        if not az_tuple[0] <= tr.stats.sac['az'] <= az_tuple[1]:
            st.remove(tr)

    return st

def bin_filter(st,bin_lat0,bin_lon0,bin_radius):
   '''
   Removes traces which lie outside of a circular bin.
   bin_radius must be given in degrees.
   '''

   for tr in st: 
      dist = gps2dist_azimuth(tr.stats.sac['stla'],tr.stats.sac['stlo'],
                              bin_lat0, bin_lon0)
      dist_m   = dist[0]
      dist_deg = kilometer2degrees(dist_m/1000.0)

      if dist_deg > bin_radius:
         st.remove(tr)

def three_component_filter(st_e, st_n, st_z):
   '''
   Takes three streams (one of each channel), and checks if each stream
   contains the same stations.  Also, checks that all traces in the stream
   have the same start time, end time, and number of samples.  If not, the
   traces will be trimmed. 
   '''

   st_e.sort()
   st_n.sort()
   st_z.sort()

   #read an initial start and end time
   t_0 = st_e[0].starttime 
   t_e = st_e[0].endtime
   t_0 += 10.0
   t_e -= 10.0

   #trim each trace to new specifications
   for tr_e in st_e:
      tr_e.trim(starttime=t_0, endtime=t_e)
   for tr_n in st_n:
      tr_n.trim(starttime=t_0, endtime=t_e)
   for tr_z in st_z:
      tr_z.trim(starttime=t_0, endtime=t_e)

def time_filter(st):
   '''
   Filters all traces in stream so that they start and stop at the 
   exact same time (sometimes they are off by a couple samples)
   '''
   #resample all data to 20.0 Hz
   for tr in st:
      tr.resample(20.0)

   #read an initial start and end time
   t_0 = st[0].stats.starttime 
   t_e = st[0].stats.endtime
   t_0 += 10.0
   t_e -= 10.0
   #trim each trace to new specifications
   for tr in st:
      tr.trim(starttime=t_0, endtime=t_e)

def lat_lon_filter(st,latmin,latmax,lonmin,lonmax):
   '''
   removes stations from a stream if the are not in specified latitude 
   and longitude range
   '''

   for tr in st:
      if (tr.stats.sac['stla'] >= latmin and
          tr.stats.sac['stla'] <= latmax and
          tr.stats.sac['stlo'] >= lonmin and
          tr.stats.sac['stla']):
         continue
      else:
         st.remove(tr)

def network_filter(st,network_name):
   '''
   removes traces from stream if the network doesn't match the specified
   network
   '''

   for tr in st:
      if tr.stats.network != network_name:
         st.remove(tr)

def equator_filter(st):
   '''
   removes traces that don't lie along the equator. (mainly useful for processing
   ses3d synthetics)
   '''
   for tr in st:
      d = abs(tr.stats.sac['stla'])
      if d >= 0.1:
         st.remove(tr)
