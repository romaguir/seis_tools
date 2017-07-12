#The following two lines are for making plots on etude compute nodes
import matplotlib as mpl
mpl.use('Agg')

import os
import h5py
import copy
import subprocess
import numpy as np
from ast import literal_eval
from scipy import interpolate
from matplotlib import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.core.util.geodetics import calcVincentyInverse
#from obspy.core.util.geodetics import gps2dist_azimuth
from obspy.core.util.geodetics import kilometer2degrees
from seis_tools.ses3d.rotation import rotate_coordinates
from scipy.signal import iirfilter, lfilter, freqz
from obspy.taup import TauPyModel
from geopy.distance import VincentyDistance
from seis_tools.seispy.misc import find_rotation_angle,find_rotation_vector
import geopy
import glob
import matplotlib.pyplot as plt

try:
    from obspy.geodetics import gps2dist_azimuth
except ImportError:
    from obspy.core.util.geodetics import gps2DistAzimuth as gps2dist_azimuth
try:
   from obspy.geodetics import kilometer2degrees
except ImportError:
   from obspy.core.util.geodetics import kilometer2degrees

def make_station_list(param_dict,plot=True,**kwargs):
   '''
   Writes, and returns list of station locations (lon,lat)

   params------------------------------------------------------------------------
   param_dict: parameter dictionary (read from file 'inparam_tomo')

   returns: station list (lon,lat)
   '''
   latmin = param_dict['latmin']
   latmax = param_dict['latmax']
   lonmin = param_dict['lonmin']
   lonmax = param_dict['lonmax']
   geometry = param_dict['station_geometry']
   dlat = param_dict['dlat']
   dlon = param_dict['dlon']
   nstations = param_dict['nstations']

   #generate random station locations
   if geometry=='random':
      lons = ((lonmax-lonmin)*np.random.random_sample(nstations)) + lonmin
      lats = ((latmax-latmin)*np.random.random_sample(nstations)) + latmin

   #generate grid of stations
   elif geometry=='grid':
      lons_ = np.arange(lonmin,lonmax+1*dlon,dlon)
      lats_ = np.arange(latmin,latmax+1*dlat,dlat)
      lons, lats = np.meshgrid(lons_,lats_)
      lons = lons.flatten()
      lats = lats.flatten()

   stations = np.array((lons,lats))

   #write station file
   f = open('station_list','w')
   for i in range(0,len(lons)):
      f.write('{} {}'.format(lons[i],lats[i])+'\n')

   #plot
   if plot:
      plt.scatter(lons,lats)
      plt.show()

   return stations

def make_earthquake_list(param_dict,**kwargs):
   '''
   Generate earthquakes to be used in tomographic inversion. Earthquake locations can 
   be generated randomly within the distance range (deltamin, deltamax), or alternatively
   a ring of earthquakes at a fixed distance from the point (0,0) can be generated.
   In the futures, events may be given as an obspy event catalog.

   args--------------------------------------------------------------------------
   param_dict: parameter dictionary (read from file 'inparam_tomo')
   
   kwargs------------------------------------------------------------------------
   nevents: number of earthquakes (only use if geometry is 'random')
   deltamin = minimum earthquake distance from (0,0) (only if geometry is 'random')
   deltamax = maximum earthquake distance from (0,0) (only if geometry is 'random')
   ringdist = distance of ring from (0,0). Can be tuple for multiple rings. (only if geometry is 'ring')
   dtheta = spacing between earthquakes in ring, given in degrees. Default = 30.
   '''
   geometry = param_dict['event_geometry']
   nevents = param_dict['nevents']
   depth = param_dict['depth']
   deltamin = param_dict['deltamin']
   deltamax = param_dict['deltamax']
   ringdist = param_dict['ringdist']
   dtheta = param_dict['dtheta']

   lat0 = kwargs.get('lat0',0.0)
   lon0 = kwargs.get('lon0',0.0)

   eq_list = []
   n = 1

   if geometry=='random':
      while len(eq_list) < nevents:
         lon = (2.0*deltamax*np.random.random(1)) - deltamax
         lat = (2.0*deltamax*np.random.random(1)) - deltamax
         dist_az = gps2dist_azimuth(lat,lon,lat0,lon0,a=6371000.0,f=0.0)
         dist_km = dist_az[0]/1000.0
         dist_deg = kilometer2degrees(dist_km)

         if dist_deg >= deltamin and dist_deg <= deltamax:
            eq_list.append((n,lon[0],lat[0],depth))
            n += 1

   elif geometry=='ring':
      theta = np.arange(0,360,dtheta)
      origin = geopy.Point(0,0)
      eq_list = []

      if type(ringdist)==int or type(ringdist)==float:
         d_km = ringdist * ((6371.0*2*np.pi)/360.0)
         for i in range(0,len(theta)):
            bearing = theta[i] 
            destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
            lat = destination[0]
            lon = destination[1]
            eq_list.append((n,lon,lat,depth))
            n += 1

      elif type(ringdist==tuple):
         for r in ringdist:
            d_km = r * ((6371.0*2*np.pi)/360.0)
            for i in range(0,len(theta)):
               bearing = theta[i] 
               destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
               lat = destination[0]
               lon = destination[1]
               eq_list.append((n,lon,lat,depth))
               n += 1

   np.savetxt('earthquake_list',eq_list,fmt=['%d','%5.5f','%5.5f','%5.5f'])
   return eq_list

def read_earthquake_list(param_dict):
   filename=param_dict['events_file']
   f = np.loadtxt(filename)
   n = f[:,0]
   lon = f[:,1]
   lat = f[:,2]
   depth = f[:,3]
   eq_list = []

   for i in range(0,len(n)):
      eq_list.append((n[i],lon[i],lat[i],depth[i]))

   return eq_list

def read_station_list(param_dict):
   filename=param_dict['stations_file']
   f = np.loadtxt(filename)
   lons = f[:,0]
   lats = f[:,1]
   stations = np.array((lons,lats))

   return stations

def read_inparam(inparam_file):
   period_list = []
   with open(inparam_file) as f:
      #read run info
      line = f.readline()
      line = f.readline()
      line = f.readline()

      line = f.readline()
      run_name = line.strip().split()[1]
      line = f.readline()
      delays_file = line.strip().split()[1]
      line = f.readline()
      phase = line.strip().split()[1]
      phases_list = phase.split(',')
      line = f.readline()
      nperiods = line.strip().split()[1]

      line = f.readline()
      periods = line.strip().split()[1]
      period_list = periods.split(',')
      line = f.readline()
      taup_model = line.strip().split()[1]
      line = f.readline()
      ray_theory = line.strip().split()[1]
      line = f.readline()
      t_sig = line.strip().split()[1]
      line = f.readline()
      add_noise = line.strip().split()[1]
      line = f.readline()
      inv_param = line.strip().split()[1]
      line = f.readline()

      #read event info
      line = f.readline()
      line = f.readline()
      event_geometry = line.strip().split()[1]
      line = f.readline()
      events_file = line.strip().split()[1]
      line = f.readline()
      nevents = line.strip().split()[1]
      line = f.readline()
      deltamin = line.strip().split()[1]
      line = f.readline()
      deltamax = line.strip().split()[1]
      line = f.readline()
      ringdist = line.strip().split()[1]
      line = f.readline()  
      dtheta = line.strip().split()[1]
      line = f.readline()
      depth = line.strip().split()[1]
      line = f.readline()

      #read station info
      line = f.readline()
      line = f.readline()
      station_geometry = line.strip().split()[1]
      line = f.readline()
      stations_file = line.strip().split()[1]
      line = f.readline()
      nstations = line.strip().split()[1]
      line = f.readline()
      latmin = line.strip().split()[1]
      line = f.readline()
      latmax = line.strip().split()[1]
      line = f.readline()
      lonmin = line.strip().split()[1]
      line = f.readline()
      lonmax = line.strip().split()[1]
      line = f.readline()
      dlat = line.strip().split()[1]
      line = f.readline()
      dlon = line.strip().split()[1]
      line = f.readline()

      #find nevents for 'ring' geometry option
      #ringdist = literal_eval(ringdist)
      #if event_geometry=='ring':
      #   if type(ringdist)==int or type(ringdist)==float:
      #      nevents = 360.0 / literal_eval(dtheta)
      #   elif type(ringdist)==tuple:
      #      nevents = len(ringdist) * (360 / literal_eval(dtheta))

      param_dict = {'run_name':run_name,
                    'delays_file':delays_file,
                    'phase':phase,
                    'nperiods':nperiods,
                    'period_list':period_list,
                    'taup_model':taup_model,
                    'event_geometry':event_geometry,
                    'nevents':int(nevents),
                    'deltamin':float(deltamin),
                    'deltamax':float(deltamax),
                    'ringdist':literal_eval(ringdist),
                    'dtheta':float(dtheta),
                    'station_geometry':station_geometry,
                    'nstations':int(nstations),
                    'latmin':float(latmin),
                    'latmax':float(latmax),
                    'lonmin':float(lonmin),
                    'lonmax':float(lonmax),
                    'dlat':float(dlat),
                    'dlon':float(dlon),
                    't_sig':literal_eval(t_sig),
                    'add_noise':literal_eval(add_noise),
                    'ray_theory':literal_eval(ray_theory),
                    'phases_list':phases_list,
                    'events_file':events_file,
                    'stations_file':stations_file,
                    'depth':literal_eval(depth),
		    'inv_param':inv_param}

      return param_dict

def rotate_delays(lat_r,lon_r,lon_0=0.0,lat_0=0.0,degrees=0):
   '''
   Rotates the source and receiver of a trace object around an
   arbitrary axis.
   '''

   alpha = np.radians(degrees)
   colat_r = 90.0-lat_r
   colat_0 = 90.0-lat_0

   x_r = lon_r - lon_0
   y_r = colat_0 - colat_r

   #rotate receivers
   lat_rotated = 90.0-colat_0+x_r*np.sin(alpha) + y_r*np.cos(alpha)
   lon_rotated = lon_0+x_r*np.cos(alpha) - y_r*np.sin(alpha)

   return lat_rotated, lon_rotated

def plot_geo_config(stations,events):
   '''
   plot geometrical configuration of tomography experiment
   '''
   m = Basemap(projection='hammer',lon_0=0,resolution='l')
   m.drawcoastlines()
   m.drawmeridians(np.arange(0,361,30))
   m.drawparallels(np.arange(-90,91,30))
   station_lons = stations[0,:]
   station_lats = stations[1,:]
   x,y = m(station_lons,station_lats)
   m.scatter(x,y,s=50,marker='^',c='r')

   for event in events:
      lon = event[1]
      lat = event[2]
      x,y = m(lon,lat)
      m.scatter(x,y,s=100,marker='*',c='y')

   plt.savefig('geo_config.pdf',format='pdf')

def get_event_params(eq_lat,eq_lon):
   dist_az = gps2dist_azimuth(eq_lat,eq_lon,0,0,a=6371000.0,f=0.0)
   dist_km = dist_az[0]/1000.0
   dist_deg = kilometer2degrees(dist_km)
   az = dist_az[1]
   baz = dist_az[2]
   rotation_angle = -1.0*((baz-180) -90.0)
   #rotation_angle = -1.0*(az-90.0)

   return dist_deg,rotation_angle

def get_filter_params(delays_file,phase,Tmin,**kwargs):
   '''
   returns the filter parameters used for cross correlation delay measurements

   args:
   delays_file: h5py delay dataset (path to file)
   phase: 'P' or 'S'
   Tmin: minimum period (string, e.g., '10.0')
   '''
   filter_type = kwargs.get('filter_type','none')
   if os.path.isfile(delays_file):
      f = h5py.File(delays_file)

   key_list = f[phase].keys()
   data = f[phase][key_list[0]][Tmin]

   if filter_type == 'none':
      filter = data.attrs['filter']
   else:
      filter = filter_type

   #ghetto fix to key problem with an h5py file
   try:
      freqmin = data.attrs['freqmin']
      freqmax = data.attrs['freqmax']
      window = data.attrs['window']
   except KeyError:
      freqmin = 1/10.0
      freqmax = 1/25.0
      window = np.array([-20.0,20.0])

   return filter,freqmin,freqmax,window

def get_filter_freqs(filter_type,freqmin,freqmax,sampling_rate,**kwargs):
   '''
   Returns frequency band information of the filter used in cross
   correlation delay time measurements.

   args-------------------------------------------------------------------------- 
   filter_type: type of filter used (e.g., bandpass, gaussian etc...)
   freqmin: minimum frequency of filter (in Hz)
   freqmax: maximum frequency of filter (in Hz)
   sampling_rate: sampling rate of seismograms (in Hz)
   kwargs-------------------------------------------------------------------------
   corners: number of corners used in filter (if bandpass). default = 2
   std_dev: standard deviation of gaussian filter (in Hz)
   plot: True or False

   returns-----------------------------------------------------------------------
   omega: frequency axis (rad/s)
   amp: frequency response of filter
   '''
   plot = kwargs.get('plot',False)
   corners = kwargs.get('corners',4)
   std_dev = kwargs.get('std_dev',0.1)
   mid_freq = kwargs.get('mid_freq',1/10.0)
   
   if filter_type == 'bandpass':
      nyquist = 0.5 * sampling_rate
      fmin = freqmin/nyquist
      fmax = freqmax/nyquist
      b, a = iirfilter(corners, [fmin,fmax], btype='band', ftype='butter')
      freq_range = np.linspace(0,0.15,200)
      w, h = freqz(b,a,worN=freq_range)
      omega = sampling_rate * w
      omega_hz = (sampling_rate * w) / (2*np.pi)
      amp = abs(h)
      
   elif filter_type == 'gaussian':
      fmin=freqmin
      fmax=freqmax
      omega_hz = np.linspace(0,0.5,200)
      omega = omega_hz*(2*np.pi)
      f_middle_hz = (fmin+fmax)/2.0
      f_middle = f_middle_hz*(2*np.pi)
      #f_middle_hz = mid_freq  #f_middle_hz = 10.0 was used in JGR manuscript
      #f_middle = f_middle_hz*(2*np.pi)
      print f_middle_hz,f_middle
      amp = np.exp(-1.0*((omega-f_middle)**2)/(2*(std_dev**2)))
      amp = amp/np.max(amp)

   if plot:
      fig,axes = plt.subplots(2)
      axes[0].plot(omega,amp)
      axes[0].set_xlabel('frequency (rad/s)')
      axes[0].set_ylabel('amplitude')
      axes[1].plot(omega_hz,amp)
      axes[1].set_xlabel('frequency (Hz)')
      axes[1].set_ylabel('amplitude')
      axes[1].axvline(fmin)
      axes[1].axvline(fmax)
      plt.show()

   return omega, amp

def make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,plot=True,**kwargs):
   '''
   find delay interpolated delay time for an arbitrary station and earthquake location

   args--------------------------------------------------------------------------
   eq_lat: earthquake latitude
   eq_lon: earthquake longitude
   station_location
   phase: 'P' or 'S'
   delays_file: h5py file containing plume travel time delays
   Tmin: Minimum period of filter

   kwargs------------------------------------------------------------------------
   lats_i: latitiude range to be used for event map (default = -30,30)
   lons_i: longitude range to be used for event map (default = -30,30)
   plot: basemap plot of delays
   return_axis: whether or not to return basemap axis
   '''
   plot = kwargs.get('plot',False)
   return_axis = kwargs.get('return_axis',False)
   lats_i = kwargs.get('lats_i',np.arange(-30.0,30.0,0.1))
   lons_i = kwargs.get('lons_i',np.arange(-30.0,30.0,0.1))
   nevent=kwargs.get('nevent',1)

   debug = False

   #-----------------------------------------------------------------------------
   #Step 1) Find earthquake distance and back azimuth (from [0,0])
   #-----------------------------------------------------------------------------
   dist_deg, rotation_angle = get_event_params(eq_lat,eq_lon)

   '''
   if dist_deg <= 35.0:
      print dist_deg
      raise ValueError('The earthquake is too close')
   #elif dist_deg >= 80.0:
   elif dist_deg >= 1300.0:
      print dist_deg
      raise ValueError('The earthquake is too far')
   '''
   if phase == 'P' or phase == 'S' or phase == 'SSS':
       event_list = ['30','40','50','60','70','80','90']
       zi = np.array((30,40,50,60,70,80,90))
   elif phase == 'SKS':
       event_list = ['70','80','90','100','110','120']
       zi = np.array((70,80,90,100,110,120))

   if debug:
      print "distance between event and (0,0) ", dist_deg
      print "make_event_delay_map: phase,event_list = ",phase,event_list

   #-----------------------------------------------------------------------------
   #Step 2) Create interpolated delay map for given distance
   #-----------------------------------------------------------------------------
   #event_list = ['35','50','65','80'] #distances of modelled events
   delay_points = []
   delay_maps = []
   
   if os.path.isfile(delays_file):
      f = h5py.File(delays_file)
   else:
      print 'delay file not found: supply absolute path to delay file'

   try:
      phase_delays = f[phase]
   except KeyError:
      print 'Phase delays for ', phase, ' dont exist'

   for event in event_list:
      try:
         delay_points.append(phase_delays[event][Tmin].value)
      except KeyError:
         print 'Delays for Tmin = ', Tmin, ' dont exist. Try one of ', phase_delays[event].keys()

   if debug:
      print 'events available for phase ',phase,' are ',phase_delays.keys()

   #get point information

   #xi = np.arange(-30,30.0,0.1) #list of x values to interpolate on
   #yi = np.arange(-30,30.0,0.1) #list of y values to interpolate on
   count = 0
   for points in delay_points:
      x = points[0]
      y = points[1]

      #rotate by back azimuth
      y,x = rotate_delays(y,x,0.0,0.0,rotation_angle)

      delays = points[2]

      #if debug:
      #   plt.scatter(x,y,c=delays)
      #	  plt.colorbar()
      #	  plt.show()
      #   plt.title(event_list[count])
      #   count += 1

      dti = mlab.griddata(x,y,delays,lons_i,lats_i,interp='linear')
      dti = np.nan_to_num(dti)
      delay_maps.append(dti.data)

   #zi = np.array((35,50,65,80))
   delay_array = np.array(delay_maps) 
   event_interpolator = interpolate.RegularGridInterpolator((zi,lats_i,lons_i),delay_array)

   #interpolate new map for given distance
   xx,yy = np.meshgrid(lons_i,lats_i)
   event_map = event_interpolator((dist_deg,yy,xx)) 

   #if debug:
   plot=True
   if plot:
      plt.clf()
      lons,lats = np.meshgrid(lons_i,lats_i)
      m = Basemap(projection='hammer',lon_0=0,resolution='l')
      m.drawcoastlines()
      m.drawmeridians(np.arange(0,351,30))
      m.drawparallels(np.arange(-90,91,30))
      lons2,lats2 = m(lons,lats)
      m.pcolormesh(lons2,lats2,event_map,vmin=-3.0,vmax=0.1)
      m.colorbar()
      lon_eq,lat_eq = m(eq_lon,eq_lat)
      m.scatter(lon_eq,lat_eq,marker='*',s=100,c='y',zorder=99)
      m.drawgreatcircle(eq_lon,eq_lat,0,0,linewidth=2,color='b')
      plt.title('{} delays at {} s'.format(phase,Tmin))
      plt.savefig('{}_{}_eq{}.png'.format('eventmap',phase,nevent))

      #if not return_axis:
      #   plt.show()

   if return_axis: 
      return event_map,m
   else:
      return event_map

def get_station_delays(event_map,stations,lats_i,lons_i,**kwargs):
   '''
   takes an event delay map and interpolates station delays
    
   args--------------------------------------------------------------------------
   event_map: delay time map for the specified earthquake
   stations: either an array of [lats,lons] or an obspy network.  if using an
             an obspy network, set the kwarg 'obspy_network' to True
   lats_i: the latitudes used in creating the event map
   lons_i: the longitudes used in creating the event map

   kwargs------------------------------------------------------------------------
   obspy_network: True if 'stations' is an obspy network object.  (default False)
   pass_figure_axis: Set to true if you wish to supply an axis for plotting
   figure_axis: the axis to plot on
   '''
   obspy_network = kwargs.get('obspy_network',False)
   pass_figure_axis = kwargs.get('pass_figure_axis',False)
   figure_axis = kwargs.get('figure_axis','none')

   if obspy_network:
      lats = []
      lons = []
      for station in stations:
         lats.append(station.latitude) 
         lons.append(stations.longitude)
   else:
      lons = stations[0,:]
      lats = stations[1,:]

   #delay_interpolator = interpolate.RegularGridInterpolator((lons_i,lats_i),event_map)

   #RM 3/31/17: I changed removed the bounds error on the interpolator...
   #            This is dangerous and I'm not sure how things will be effected.
   #            This is for preliminary testing of the Pacific array geometry
   delay_interpolator = interpolate.RegularGridInterpolator((lons_i,lats_i),event_map,
                                                            bounds_error=False,fill_value=0.0)
   station_delays = delay_interpolator((lats,lons))

   if pass_figure_axis:
      lons_bsmp, lats_bsmp = figure_axis(lons,lats)
      figure_axis.scatter(lons_bsmp,lats_bsmp,marker='^',s=50,c=station_delays,vmin=-3.0,vmax=0.1)
      plt.show()

   return station_delays
   

def write_input(eq_lat,eq_lon,eq_dep,ievt,stations,phase,delays_file,Tmin,taup_model,filename,raytheory=False,tt_from_raydata=True,**kwargs):
   '''
   write an input file for globalseis finite frequency tomography software.
   each earthquake and datatype (P,S,etc...) has it's own input file

   args--------------------------------------------------------------------------
   eq_lat: earthquake latitude (deg)
   eq_lon: earthquake longitude (deg)
   eq_dep: earthquake depth (km)
   stations: stations array (lons,lats)
   delays_file: h5py datafile containing cross correlation delay times
   Tmin: minimum period at which cross correlation measurements were made
   taup_model: name of TauPyModel used to calculate 1D travel times
   filename:
   raytheory: True or False
   tt_from_raydata: If True, writes cross correlation times to 'xcor*', which 
                    will then be added to 1D travel times from raydata

   kwargs------------------------------------------------------------------------
   plot_figure: plot a figure showing source receiver geometry and delay map
   t_sig: estimated standard error in cross correlation measurement.
   add_noise: add gaussian noise to traveltime measurements of magnitude t_sig
   fake_SKS_header: test the SKS header
   '''
   #define variables used in finite frequency tomography (kwargs)----------------
   idate = kwargs.get('idate','15001') #event date YYDDD where DDD is between 1 and 365
   iotime =  kwargs.get('iotime','010101') #vent origin time (HHMMSS)
   kluster = kwargs.get('kluster','0') #0 if no clustering used
   stationcode = kwargs.get('stationcode','XXXX') #station code (no more than 16 chars)
   netw = kwargs.get('netw','PLUMENET ') #network code
   nobst = kwargs.get('nobst','1') #number of travel time measurements
   nobsa = kwargs.get('nobsa','0') #number of amplitude measurements
   kpole = kwargs.get('kpole','0') #number of polar crossings (0 for P and S)
   sampling_rate = kwargs.get('sampling_rate',10.0)
   n_bands = kwargs.get('n_bands',1) # spectral bands used (TODO setup more than one)
   kunit = kwargs.get('kunit',1) #unit of noise (1 = nm)
   rms0 = kwargs.get('rms0',0) #don't know what this is
   plot_figure = kwargs.get('plot_figure',False)
   dist_min = kwargs.get('dist_min',30.0)
   dist_max = kwargs.get('dist_max',90.0)
   t_sig = kwargs.get('t_sig',0.0)
   add_noise = kwargs.get('add_noise',False)
   fake_SKS_header = kwargs.get('fake_SKS_header',False)
   filter_type = kwargs.get('filter_type','none')

   ievt=int(ievt) #double check ievt is an integer (in case it was read from a file)

   debug = False

   #create taup model------------------------------------------------------------
   tt_model = TauPyModel(taup_model)

   #get filter parameters--------------------------------------------------------
   print 'Tmin = ', Tmin
   filter_type, freqmin,freqmax, window = get_filter_params(delays_file,phase,Tmin,filter_type=filter_type)
   omega,amp =  get_filter_freqs(filter_type,freqmin,freqmax,sampling_rate)
   window_len = window[1] - window[0]

   #write header-----------------------------------------------------------------
   f = open(filename,'w')
   f.write('{}'.format(filename)+'\n')
   f.write('{}'.format('None'+'\n'))
   fdelays = open('xcor_{}'.format(filename),'w')

   #ray information--------------------------------------------------------------
   if phase == 'P':
      gm_component = 'BHZ ' #ground motion component
      f.write('P'+'\n')
      f.write('P'+'\n')
      f.write('6371 1 1'+'\n')
      f.write('3482 2 1'+'\n')
      f.write('6371 5 0'+'\n')
   elif phase == 'S' and fake_SKS_header == False:
      gm_component = 'BHT ' #ground motion component
      f.write('S'+'\n')
      f.write('S'+'\n')
      f.write('6371 1 2'+'\n')
      f.write('3482 2 2'+'\n')
      f.write('6371 5 0'+'\n')
   elif phase == 'SKS' or fake_SKS_header == True:
      gm_component = 'BHR ' #ground motion component
      f.write('SKS'+'\n')
      f.write('SKS'+'\n')
      f.write('6371 1 2'+'\n')
      f.write('3482 4 1'+'\n')
      f.write('1217.1 2 1'+'\n')
      f.write('3482 4 2'+'\n')
      f.write('6371 5 0'+'\n')

   #this is hardwired for now (based on range of rays found with ray tracing software)
   #TODO make distance range more adaptable
   if phase == 'P':
      dist_min = 30.0
      #dist_max = 98.3859100
      dist_max = 97.0
   elif phase == 'S':
      dist_min = 30.0
      #dist_max = 99.0557175
      dist_max = 97.0
   elif phase == 'SKS':
      #dist_min = 66.0320663
      #dist_max = 144.349365
      dist_min = 68.0
      dist_max = 142.0

   #write spectral band information-----------------------------------------------
   if raytheory:
      n_bands=0
      f.write('{}'.format(n_bands)+'\n')
   else:
      f.write('{}'.format(n_bands)+'\n')
      f.write('{}'.format(len(omega))+'\n')
      for i in range(0,len(omega)): 
         f.write('{} {}'.format(omega[i],amp[i])+'\n')

   #event delay map--------------------------------------------------------------
   #lats_i = np.arange(-30.0,30.0,0.1)
   #lons_i = np.arange(-30.0,30.0,0.1)
   lats_i = np.arange(-45.0,45.0,0.1)
   lons_i = np.arange(-45.0,45.0,0.1)

   if plot_figure:
      event_map,figure_axis = make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,lats_i=lats_i,lons_i=lons_i,plot=True,return_axis=False,nevent=ievt)
   else:
      if debug:
         print 'func:write_input- making event delay map for', phase
         #print 'eq_lat,eq_lon,phase,Tmin lats_i,lons_i',eq_lat,eq_lon,phase,Tmin,lats_i,lons_i
      event_map = make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,lats_i=lats_i,                                          lons_i=lons_i,return_axis=False,plot=True,nevent=ievt)

   #find delays at stations------------------------------------------------------
   if plot_figure:
      station_delays = get_station_delays(event_map,stations,lats_i,lons_i,pass_figure_axis=True,figure_axis=figure_axis)
   else: 
      station_delays = get_station_delays(event_map,stations,lats_i,lons_i)

   #add noise (optional)---------------------------------------------------------
   if t_sig != 0:
      noise = np.random.normal(0,t_sig,len(station_delays))
      if add_noise:
         station_delays += noise

   station_lons = stations[0,:]
   station_lats = stations[1,:]
   n_stations = len(station_lats)
   station_elevation = 0.0

   for i in range(0,n_stations):
      dist_deg, rotation_angle = get_event_params(eq_lat,eq_lon)
      #find event distance
      event_distaz = gps2dist_azimuth(eq_lat,eq_lon,station_lats[i],station_lons[i],a=6371000.0,f=0.0)
      event_dist_deg = kilometer2degrees((event_distaz[0]/1000.0)) 

      #skip station if too close or too far from source
      if event_dist_deg <= dist_min or event_dist_deg >= dist_max:
         continue

      #get ray theoretical travel time 
      #if phase == 'S':
      #   phase_list = ['s','S','Sdiff']
      #elif phase == 'P':
      #   phase_list = ['p','P','Pdiff']

      ray_theory_arr = tt_model.get_travel_times(eq_dep,event_dist_deg,phase_list=[phase])

      ### TRY TO GET TRAVEL TIME IN CORE #########################################################
      ray_theory_path = tt_model.get_ray_paths(eq_dep,event_dist_deg,phase_list=[phase])
      phase_path = ray_theory_path[0]
      path_time = phase_path.path['time']
      path_dt = np.diff(path_time)
      path_depth = phase_path.path['depth']
      time_in_core = 0
      for p_i in range(0,len(path_dt)):
         if path_depth[p_i] >= 2889.0:
            time_in_core += path_dt[p_i] 
      ############################################################################################
      if debug:
	 print '_________________________________________________________________________________'
         print 'arrivals from taup_get_travel_time for event parameters [depth,delta(deg),phase]:'
	 print '[{},{},{}]'.format(eq_dep,event_dist_deg,phase),ray_theory_arr
	 print 'time in core: ', time_in_core
	 print '_________________________________________________________________________________'

      ray_theory_travel_time = ray_theory_arr[0].time
      delay_time = station_delays[i] 
      tobs = ray_theory_travel_time - delay_time

      if debug:
          print 'distance, phase, raytheory travel time, observed delay:', event_dist_deg,phase,ray_theory_travel_time,delay_time
          print 'the travel time observation is ', tobs

      fdelays.write('{}'.format(delay_time)+'\n')

      if raytheory:
         n_bands = 0
         nbt = 0 #spectral band number (must be 0 if ray theory)
         window_len = 0
         kunit = 0
         corcoeft=0
      else:
         nbt = 1 #spectral band number

      #write line 1--------------------------------------------------------------
      f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(idate,
              iotime,ievt,kluster,stationcode,netw,gm_component,eq_lat,eq_lon,eq_dep,
              station_lats[i],station_lons[i],station_elevation,nobst,nobsa,kpole)+'\n')

      #write line 2--------------------------------------------------------------
      f.write('{} {} '.format(kunit,rms0))
      for j in range(0,n_bands+1):
         f.write('0')   #used to be 0.0 
      f.write('\n')

      #write line 3---------------------------------------------------------------
      if raytheory:
         f.write('{}'.format(1)+'\n')
      else:
         f.write('{}'.format(n_bands)+'\n')
      
      #write line 4---------------------------------------------------------------
      corcoeft = 1.0 # cross correlation coefficient 
      f.write('{} {} {} {} {} {} {}'.format(tobs,t_sig,corcoeft,nbt,window_len,time_in_core,'#tobs,tsig,corcoeft,nbt,window,tincore')+'\n') 

      #write line 5--------------------------------------------------------------
      f.write('{}'.format(0)+'\n')

def write_inmpisolve(run_name='inversion',**kwargs):
   chi2 = kwargs.get('chi2',1.0)
   ksmooth = kwargs.get('ksmooth',1)
   epsnorm = kwargs.get('epsnorm',1.0)
   epssmooth = kwargs.get('epssmooth',0)
   epsratio = kwargs.get('epsratio',0.2)
   vertsm = kwargs.get('vertsm',5)
   outlier_magnitude = kwargs.get('outlier_magnitude',20.0)
   max_iter = kwargs.get('max_iter',50)
   root = kwargs.get('root',250)

   f = open('in.mpisolvetomo','w')
   f.write(os.getcwd()+'\n')
   f.write(run_name+'\n')
   f.write(run_name+'\n')
   f.write('{} {} {} {} {} {}'.format(chi2, ksmooth, epsnorm, epssmooth, epsratio, vertsm))
   f.write(' #chi^2/N, ksmooth, epsnorm, epssmooth, epsratio \n')
   f.write('{}'.format(outlier_magnitude))
   f.write(' #max magnitude of outliers allowed \n')
   f.write('{} {}'.format(max_iter,root))
   f.write(' #max number of iterations during search, rootfinding')

def write_mpisubmit():
   executable = '/geo/home/romaguir/globalseis/bin/mpisolvetomov'
   f = open('submit_mpisolve.qsub','w')
   f.write('#/bin/bash \n')
   f.write('#PBS -N mpisolvetomo \n')
   f.write('#PBS -q default \n')
   f.write('#PBS -l nodes=2:ppn=8,walltime=20:00:00 \n')
   f.write('#PBS -d . \n')
   f.write('#PBS -V \n')
   f.write('mpirun -np 16 '+executable+' < in.mpisolvetomo > out.mpisolvetomo \n')
   f.write('rm mat*')

def write_doit(param_dict,run_inversion=False,tt_from_raydata=True):
   assemblematrix_exe = '/geo/home/romaguir/globalseis/bin/assemblematrixv'
   f = open('doit','w')
   f.write('#!/bin/bash \n')
   f.write('{} {}'.format('runraydata',param_dict['run_name'])+'\n')
   if tt_from_raydata:
      f.write('/geo/home/romaguir/globalseis/bin/tt_from_raydata \n')
      f.write('{} {}'.format('runraydata',param_dict['run_name'])+'\n')
   if param_dict['inv_param'] == 'Vp':
      f.write('{} {} {}'.format('runvoxelmatrix',param_dict['run_name'],'P') +'\n')
   elif param_dict['inv_param'] == 'Vs':
      f.write('{} {} {}'.format('submit_voxelmatrix',param_dict['run_name'],' etude') +'\n')
   elif param_dict['inv_param'] == 'Qs':
      f.write('{} {} {}'.format('submit_voxelmatrix ',param_dict['run_name'],'Qs') +'\n')
   if param_dict['inv_param'] == 'Vp':
      f.write('{} {} {}'.format('runassemblematrix','P','inversion') +'\n')
   elif param_dict['inv_param'] == 'Vs':
      f.write('{} {} {}'.format('runassemblematrix','S','alldata') +'\n')
   elif param_dict['inv_param'] == 'Qs':
      f.write('{} {} {}'.format('runassemblematrix','Qs','alldata') +'\n')
   if run_inversion:
      f.write('qsub submit_mpisolve.qsub \n')

   subprocess.call('chmod +x doit',shell=True)

def rotate_earthquake_list(earthquake_list,s1,s2,filename):
   f = np.loadtxt(earthquake_list)
   event_num = f[:,0]
   lon = f[:,1]
   lat = f[:,2]
   depth = f[:,3]
   colat = 90.0 - lat
   colat_rotated = np.zeros(len(colat))
   lon_rotated = np.zeros(len(lon))

   s1_c = copy.deepcopy(s1)
   s2_c = copy.deepcopy(s2)
   n = find_rotation_vector(s1_c,s2_c)
   s1_c = copy.deepcopy(s1)
   s2_c = copy.deepcopy(s2)
   phi = find_rotation_angle(s1_c,s2_c) 
   print n,phi

   for i in range(0,len(event_num)):
      coors = rotate_coordinates(n=n,phi=phi,colat=colat[i],lon=lon[i]) 
      colat_rotated[i] = coors[0]
      lon_rotated[i] = coors[1]

   lat_rotated = 90.0-colat_rotated

   fout = open(filename,'w')
   for i in range(0,len(event_num)):
      fout.write('{} {} {} {}'.format(i+1,lon_rotated[i],lat_rotated[i],depth[i])+'\n')

def rotate_station_list(station_list,s1,s2,filename):
   f = np.loadtxt(station_list)
   lon = f[:,0]
   lat = f[:,1]
   colat = 90.0 - lat
   colat_rotated = np.zeros(len(colat))
   lon_rotated = np.zeros(len(lon))

   s1_c = copy.deepcopy(s1)
   s2_c = copy.deepcopy(s2)
   n = find_rotation_vector(s1_c,s2_c)
   s1_c = copy.deepcopy(s1)
   s2_c = copy.deepcopy(s2)
   phi = find_rotation_angle(s1_c,s2_c) 
   print n,phi

   for i in range(0,len(lat)):
      coors = rotate_coordinates(n=n,phi=phi,colat=colat[i],lon=lon[i]) 
      colat_rotated[i] = coors[0]
      lon_rotated[i] = coors[1]

   lat_rotated = 90.0-colat_rotated

   fout = open(filename,'w')
   for i in range(0,len(lat)):
      fout.write('{} {}'.format(lon_rotated[i],lat_rotated[i])+'\n')

def plot_delays_file(delays_file,phase='S',period='10.0',ep_d='all',**kwargs):
   plot_type = kwargs.get('plot_type','imshow')
   f = h5py.File(delays_file,'r')
   seismic_phase = f[phase]
   if ep_d == 'all':
      distances = seismic_phase.keys()
      for distance in distances:
         fig_title = 'phase = {}, $\Delta = $ {}'.format(phase,distance)
         points = seismic_phase[distance][period].value
         points = points.T
         x = points[:,0]
         y = points[:,1]
         dT = points[:,2]
         l = np.sqrt(len(x))
         dT = dT.reshape(l,l)
         if plot_type == 'scatter':
            plt.scatter(points[:,0],points[:,1],c=points[:,2])
         elif plot_type == 'imshow':
            plt.imshow(dT,extent=(x.min(),x.max(),y.min(),y.max()))
         plt.colorbar()
         plt.title(fig_title) 
         plt.show()
   else:
      distance = ep_d
      fig_title = 'phase = {}, $\Delta = $ {}'.format(phase,distance)
      points = seismic_phase[distance][period].value
      points = points.T
      x = points[:,0]
      y = points[:,1]
      dT = points[:,2]
      l = np.sqrt(len(x))
      dT = dT.reshape(l,l)
      if plot_type == 'scatter':
         plt.scatter(points[:,0],points[:,1],c=points[:,2])
      elif plot_type == 'imshow':
         plt.imshow(dT,extent=(x.min(),x.max(),y.min(),y.max()))
      plt.colorbar()
      plt.show()
      

def get_delays_from_file(delays_file,phase,ep_dist,period='10.0'):
   f = h5py.File(delays_file,'r')
   fmt_string = phase+'/'+ep_dist+'/'+period
   delays = f[fmt_string].value
   lons = delays[0,:]
   lats = delays[1,:]
   vals = delays[2,:]
   return lons,lats,vals
   
def edit_delays_file(delays_file,phase,ep_dist,lat,lon,new_val,period='10.0'):
   f = h5py.File(delays_file,'r+')
   fmt_string = phase+'/'+ep_dist+'/'+period
   delays = f[fmt_string].value
   lons = delays[0,:]
   lats = delays[1,:]
   vals = delays[2,:]
   print delays
   print 'MIN/MAX = ',min(vals),max(vals)
   for i in range(0,len(lons)):
      if lons[i] == lon and lats[i] == lat:
         vals[i] = new_val
         print 'CHANGED A VALUE TO ',new_val
   print 'MIN/MAX = ',min(vals),max(vals)
   new_dset = np.array((lons,lats,vals))
   del f[fmt_string]
   dset = f.create_dataset(fmt_string,data=new_dset)
   f.close()


def calculate_ray_coverage(earthquake_list,stations_list,depth_range,phase='S',**kwargs):
   '''
   args:
      earthquake_list: earthquakes file (same format as used in synth tomo)
      stations_list: stations file (same format as used in synth tomo)
      depth_range: tuple (mindepth,maxdepth)

   kwargs:
      savefig: True or False
      figname: str, name of figure (defaults to fig.pdf)
      plot_title: str, title at the top of the plot (default no title)
   '''
   #the earthquake list format is: eq num, eq lon, eq lat, eq dep
   #the stations list format is: st lon, st lat
   savefig = kwargs.get('savefig',True)
   fig_name = kwargs.get('fig_name','fig.pdf')
   plot_title = kwargs.get('plot_title','None')
   fout_name = kwargs.get('fout_name','None')
   prem = TauPyModel('prem_50km')

   stations_file = np.loadtxt(stations_list)
   quakes_file = np.loadtxt(earthquake_list)
   n_quakes = len(quakes_file)
   n_stats = len(stations_file)
   st_lons = stations_file[:,0]
   st_lats = stations_file[:,1]
   eq_lons = quakes_file[:,1]
   eq_lats = quakes_file[:,2]
   eq_deps = quakes_file[:,3]

   if phase=='S' or phase=='P':
      delta_min = 30.0
      delta_max = 100.0
   elif phase == 'SKS':
      delta_min = 70.0
      delta_max = 140.0

   m = Basemap(projection='hammer',lon_0=204)
   fout = open(fout_name,'w')

   for i in range(0,n_quakes):
      #print 'working on earthquake', i
      for j in range(0,n_stats):

          geodet = gps2dist_azimuth(eq_lats[i], eq_lons[i], st_lats[j], st_lons[j])
          dist_m = geodet[0]
          dist_deg = kilometer2degrees((dist_m/1000.))

          if dist_deg < delta_min:
             continue
          elif dist_deg > delta_max:
             continue

          az = geodet[1]

          #print 'eq_lat, eq_lon, st_lat, st_lon, dist_deg', eq_lats[i],eq_lons[i],st_lats[j],st_lons[j],dist_deg
          arrs = prem.get_pierce_points(source_depth_in_km=eq_deps[i],
                                        distance_in_degree=dist_deg,
                                        phase_list=[phase])
          #print arrs
          arr = arrs[0]
          pierce_dict = arr.pierce
          #items in pierce_dict: 'p' (slowness), 'time' (time in s), 'dist', (distance in rad), 'depth' (depth in km)


          origin = geopy.Point(eq_lats[i],eq_lons[i])
          bearing = az
          geo_path = []
          cross_pt1 = 0
          cross_pt2 = 0
          dist_max = pierce_dict['dist'][::-1][0]
          for ds in pierce_dict:
             #only add points that are past turning depth
             dist_here = ds[2]
             if dist_here >= dist_max / 2:
                time_here = ds[1]
                depth_here = ds[3]
                if depth_here == depth_range[1]:
                   dist_deg = np.degrees(ds[2])
                   dist_km = dist_deg * ((2*np.pi*6371.0/360.0))
                   geo_pt = VincentyDistance(kilometers=dist_km).destination(origin,bearing)
                   lat_pt = geo_pt[0]
                   lon_pt = geo_pt[1]
                   cross_pt1 = (lon_pt,lat_pt)
                if depth_here == depth_range[0]:
                   dist_deg = np.degrees(ds[2])
                   dist_km = dist_deg * ((2*np.pi*6371.0/360.0))
                   geo_pt = VincentyDistance(kilometers=dist_km).destination(origin,bearing)
                   lat_pt = geo_pt[0]
                   lon_pt = geo_pt[1]
                   cross_pt2 = (lon_pt,lat_pt)
          if cross_pt1 != 0 and cross_pt2 != 0:
             m.drawgreatcircle(cross_pt1[0],cross_pt1[1],cross_pt2[0],cross_pt2[1],linewidth=1,alpha=0.15,color='k')
             fout.write('{} {} {} {}'.format(cross_pt1[0],cross_pt1[1],cross_pt2[0],cross_pt2[1])+'\n')

   m.drawcoastlines()
   m.fillcontinents(color='lightgrey')
   #m.drawparallels(np.arange(-90.,120.,30.))
   #m.drawmeridians(np.arange(0.,360.,60.))
   if plot_title != 'None':
      plt.title(plot_title)

   if savefig:
      plt.savefig(fig_name)
      plt.clf()
   else:
      plt.show()
