import os
import subprocess
import numpy as np
from seis_tools.synth_tomo import ff_tomo

#read params---------------------------------------------------------------------
param_dict = ff_tomo.read_inparam('./inparam_tomo')

#make stations-------------------------------------------------------------------
stations = ff_tomo.make_stations(latmin=param_dict['latmin'],
                                 latmax=param_dict['latmax'],
                                 lonmin=param_dict['lonmin'],
                                 lonmax=param_dict['lonmax'],
                                 geometry=param_dict['station_geometry'],
                                 dlat=param_dict['dlat'],
                                 dlon=param_dict['dlon'],
                                 plot=False)

#make events---------------------------------------------------------------------
events = ff_tomo.make_earthquake_list(nevents=param_dict['nevents'],
                                      deltamin=param_dict['deltamin'],
                                      deltamax=param_dict['deltamax'],
                                      geometry=param_dict['event_geometry'],
                                      ringdist=param_dict['ringdist'],
                                      dtheta=param_dict['dtheta'])

#make run directory--------------------------------------------------------------
os.mkdir(param_dict['run_name'])
os.chdir(param_dict['run_name'])
subprocess.call("cp ../inparam_tomo .",shell=True)
subprocess.call("cp ../earthquake_list .",shell=True)
subprocess.call("cp ../stations_list .",shell=True)

#plot experiment geometry--------------------------------------------------------
ff_tomo.plot_geo_config(stations=stations,events=events)

#write input files---------------------------------------------------------------
for event in events:
   ievt = event[0]
   eq_lat = event[1]
   eq_lon = event[2]
   eq_dep = event[3]
   filename = '{}_{}'.format(param_dict['run_name'],ievt)

   ff_tomo.write_input(eq_lat=event[1],
                       eq_lon=event[2],
                       eq_dep=event[3],
                       ievt=event[0],
                       stations=stations, 
                       phase=param_dict['phase'],
                       delays_file=param_dict['delays_file'],
                       Tmin=param_dict['period_list'][0],
                       taup_model=param_dict['taup_model'],
                       filename=filename,
                       raytheory=param_dict['ray_theory'],
                       plot_figure=False,
                       t_sig=param_dict['t_sig'],
                       add_noise=param_dict['add_noise'])

ff_tomo.write_run_raydata(run_name=param_dict['run_name'],nevents=param_dict['nevents'])
ff_tomo.write_run_voxelmatrix(run_name=param_dict['run_name'],nevents=param_dict['nevents'],model_par=param_dict['phase'])
ff_tomo.write_run_assemblematrix(param_dict)
ff_tomo.write_inmpisolve()
ff_tomo.write_mpisubmit()
ff_tomo.write_doit(run_inversion=True)

#move crust 2.0 files to current directory---------------------------------------
subprocess.call("cp /geo/home/romaguir/utils/crust2.0/* .",shell=True)
