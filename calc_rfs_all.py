import os
import glob
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from seis_tools.seispy import *
from seis_tools.seispy.receiver_functions import receiver_function
from seis_tools.seispy.receiver_functions import write_h5py_dict
from seis_tools.seispy.receiver_functions import s2nr
from obspyh5 import set_index

#Basic options-----------------------------------------------------------------
run_name = 'wl'
#decon_method = 'damped_lstsq'
decon_method = 'water_level'
only_use_TA  = True
#filter_range = (1/50.0,1/5.0) #min and max frequencies of bandpass (Hz)
filter_range = (1/50.0,1/5.0) #min and max frequencies of bandpass (Hz)
model = TauPyModel(model='ak135')
#rf_window = [-10,100]
rf_window = [-10,120]

#Define data directory---------------------------------------------------------
data_dir = '/geo/work10/romaguir/seismology/data/US_TA/'
BHE_list = glob.glob(data_dir+'*BHE*')
BHN_list = glob.glob(data_dir+'*BHN*')
BHZ_list = glob.glob(data_dir+'*BHZ*')
BHE_list.sort()
BHN_list.sort()
BHZ_list.sort()

#TODO: make sure that each list contains what it's supposed to

traces_in_stack = 0
master_stack = 0
num_stations = 0
station_locs = []
quake_locs = []
num_events = len(BHE_list)

print 'num events = ', num_events

#for writing output to h5
set_index(index='receiver_function')
IGNORE_RF = ('sac', 'back_azimuth', 'distance', '_format','rf_id','channel','calib','location')

#loop through events-------------------------------------------------------------
for i in range(0,num_events):
   print 'reading streams for event # ',i+1,'of ',num_events, BHE_list[i]
   ste = obspy.read(BHE_list[i],format='PICKLE')
   stn = obspy.read(BHN_list[i],format='PICKLE')
   stz = obspy.read(BHZ_list[i],format='PICKLE')

   #stream pre-processing--------------------------------------------------------
   ste.sort()
   stn.sort()
   stz.sort()
   filter.range_filter(ste,(30,90))
   filter.range_filter(stn,(30,90))
   filter.range_filter(stz,(30,90))
   
   if len(ste) == 0 or len(stn) == 0 or len(stz) == 0:
      print 'all traces have been winnowed, skipping event'
      continue

   #resample
   filter.gimp_filter(ste)
   filter.gimp_filter(stn)
   filter.gimp_filter(stz)

   #ste.resample(10.0)
   #stn.resample(10.0)
   #stz.resample(10.0)
   ste.resample(5.0)
   stn.resample(5.0)
   stz.resample(5.0)

   #check each stream has same number of traces
   if len(ste) != len(stz) or len(ste) != len(stn) or len(stn) != len(stz):
      print "stream lengths not equal, skipping event: len(ste), len(stn), len(stz) ", len(ste),len(stn),len(stz)
      continue

   #normalize
   ste.normalize()
   stn.normalize()
   stz.normalize()
   #filter
   ste.filter('bandpass',
              freqmin=filter_range[0],
              freqmax=filter_range[1],
              corners=2, zerophase=True)
   stn.filter('bandpass',
              freqmin=filter_range[0],
              freqmax=filter_range[1],
              corners=2, zerophase=True)
   stz.filter('bandpass',
              freqmin=filter_range[0],
              freqmax=filter_range[1],
              corners=2, zerophase=True)
   

   #calculate receiver functions-------------------------------------------------
   print 'starting receiver function calculation for ',len(ste), 'stations'
   for j in range(0,len(ste)): 
      tre = ste[j]
      trn = stn[j]
      trz = stz[j]

      #the us array data has the back azimuth flipped 180... correct here
      if tre.stats.sac['baz'] <= 180.0:
         tre.stats.sac['baz'] += 180.0
      else:
         tre.stats.sac['baz'] -= 180.0
   
      #append station location
      station_loc = (tre.stats.sac['stlo'],tre.stats.sac['stla'])
      station_locs.append(station_loc)
 
      #append earthquake location
      quake_loc = (tre.stats.sac['evlo'],tre.stats.sac['evla'])
      quake_locs.append(quake_loc)


      #get receiver function
      rf = receiver_function(tre,trn,trz,taup_model=model,window=[-10,120])
      rf.get_prf(rotation_method='RTZ',decon_type=decon_method)


      #get signal to noise ratio
      snr = s2nr(rf.rf_st[1])

      #zero pP and PP
      rf.zero_pP()
      rf.zero_PP()

      #write raw receiver function to h5py file
      rf_out = write_h5py_dict(rf.rf_st[1],signal_to_noise_ratio=snr,rf_id=i,ray_param=rf.ray_param)
      print rf_out.stats
      rf_out.write('receiver_functions'+run_name+'.h5','H5',mode='a',ignore=IGNORE_RF)

      #moveout correction
      rf.moveout_correction()

      #stack
      master_stack += rf.rf_st[1].data
      traces_in_stack += 1

#write output--------------------------------------------------------------------
station_data = list(set(station_locs))
logfile      = open('logfile_'+run_name,'w')
logfile.write('number of traces: '+str(traces_in_stack)+'\n')
logfile.write('number of events: '+str(num_events)+'\n')
logfile.write('deconvolution method: '+decon_method+'\n')
logfile.write('band pass:'+str(filter_range)+' Hz'+'\n')
stationfile = open('stations','w')
stationfile.write('STATIONS: longitude, latitude'+'\n')
for station in station_data:
   stationfile.write(str(station[0])+' '+str(station[1])+'\n')

#plot----------------------------------------------------------------------------
master_stack /= max(abs(master_stack))                                 #normalize
time = np.linspace(rf_window[0],rf_window[1],len(master_stack))
np.savetxt(fname='master_stack_'+run_name+'.dat',X=np.c_[time,master_stack])
plt.style.use('mystyle')
plt.plot(time,master_stack)
plt.savefig('master_stack'+run_name+'.png',dpi=200)

#make gmt plot-------------------------------------------------------------------------------------
#make header dictionary
#header = {'N_traces':str(traces_in_stack),'N_events':str(num_events),'decon':decon_method,'frequency':str(filter_range)+' Hz'}
#plot.gmt_north_america(station_data=station_data,quake_locs=quake_locs,rf_data=(master_stack,time),header=header)
