import os
import subprocess
import numpy as np
from seis_tools.synth_tomo import ff_tomo
from obspy.core.util.geodetics import gps2dist_azimuth
from obspy.core.util.geodetics import kilometer2degrees

#read params---------------------------------------------------------------------
param_dict = ff_tomo.read_inparam('./inparam_tomo')

#make stations-------------------------------------------------------------------
if param_dict['station_geometry'] in ['random','grid']:
   stations = ff_tomo.make_station_list(param_dict,plot=False)
elif param_dict['station_geometry'] == 'read':
   stations = ff_tomo.read_station_list(param_dict)
   subprocess.call("cp "+param_dict['stations_file']+" station_list",shell=True)

#make events---------------------------------------------------------------------
if param_dict['event_geometry'] in ['random','ring','single']:
   events = ff_tomo.make_earthquake_list(param_dict)
elif param_dict['event_geometry'] == 'read':
   events = ff_tomo.read_earthquake_list(param_dict)
   subprocess.call("cp "+param_dict['events_file']+" earthquake_list",shell=True)

#make run directory--------------------------------------------------------------
os.mkdir(param_dict['run_name'])
os.chdir(param_dict['run_name'])
subprocess.call("cp ../inparam_tomo .",shell=True)
subprocess.call("cp ../earthquake_list .",shell=True)
subprocess.call("cp ../station_list .",shell=True)

#plot experiment geometry--------------------------------------------------------
ff_tomo.plot_geo_config(stations=stations,events=events)

#write input files---------------------------------------------------------------
ievt = 1 #event number
nP = 0 #number of P observations
nS = 0 #number of S observations
nSKS = 0 #number of SKS observations

for event in events:
   ievt = event[0]
   eq_lon = event[1]
   eq_lat = event[2]
   eq_dep = event[3]

   dist_az = gps2dist_azimuth(eq_lat,eq_lon,0,0,a=6371000.0,f=0.0)
   event_dist_deg = kilometer2degrees((dist_az[0]/1000.0))

   for phase in param_dict['phases_list']:

      if phase == 'SKS':
         if event_dist_deg < 70.0 or event_dist_deg > 120.0:
             print 'skipping event for phase SKS at distance',event_dist_deg
             continue
         else:
            nSKS += 1
            filename = '{}_{}'.format(param_dict['run_name']+'.SKS',nSKS)
      elif phase == 'S': 
         if event_dist_deg < 30.0 or event_dist_deg > 90.0:
             print 'skipping event for phase ',phase,'at distance',event_dist_deg
             continue
         else:
            nS += 1
            filename = '{}_{}'.format(param_dict['run_name']+'.S',nS)

      print 'eq lon,lat,depth :', eq_lon, eq_lat, eq_dep
      ff_tomo.write_input(eq_lat=eq_lat,
                          eq_lon=eq_lon,
                          eq_dep=eq_dep,
                          ievt=ievt,
                          stations=stations, 
                          phase=phase,
                          delays_file=param_dict['delays_file'],
                          Tmin=param_dict['period_list'][0],
                          taup_model=param_dict['taup_model'],
                          filename=filename,
                          raytheory=param_dict['ray_theory'],
                          plot_figure=False,
                          t_sig=param_dict['t_sig'],
                          add_noise=param_dict['add_noise']) 
      ievt += 1

#write scripts to run ff software------------------------------------------------
ff_tomo.write_inmpisolve()
ff_tomo.write_mpisubmit()
ff_tomo.write_doit(param_dict,tt_from_raydata=True,run_inversion=False)

#move event maps to their own directory------------------------------------------
subprocess.call("mkdir eventmaps",shell=True)
subprocess.call("mv eventmap*png* eventmaps",shell=True)

#move crust 2.0 files to current directory---------------------------------------
subprocess.call("cp /geo/home/romaguir/utils/crust2.0/* .",shell=True)
