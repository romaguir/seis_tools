import os
import h5py
import subprocess
import numpy as np
from ast import literal_eval
from scipy import interpolate
from matplotlib import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.core.util.geodetics import kilometer2degrees
from scipy.signal import iirfilter, lfilter, freqz
from obspy.taup import TauPyModel
from geopy.distance import VincentyDistance
import geopy

def make_stations(latmin,latmax,lonmin,lonmax,geometry='random',plot=True,**kwargs):

   dlat = kwargs.get('dlat',2.0)
   dlon = kwargs.get('dlon',2.0)
   nstations = kwargs.get('nstations',20)

   if geometry=='random':
      lons = ((lonmax-lonmin)*np.random.random_sample(nstations)) + lonmin
      lats = ((latmax-latmin)*np.random.random_sample(nstations)) + latmin

   elif geometry=='grid':
      lons_ = np.arange(lonmin,lonmax+1*dlon,dlon)
      lats_ = np.arange(latmin,latmax+1*dlat,dlat)
      lons, lats = np.meshgrid(lons_,lats_)
      lons = lons.flatten()
      lats = lats.flatten()

   stations = np.array((lats,lons))

   f = open('stations_list','w')
   for i in range(0,len(lons)):
      f.write('{} {}'.format(lons[i],lats[i])+'\n')

   if plot:
      plt.scatter(lons,lats)
      plt.show()


   return stations

def make_earthquake_list(geometry='random',obspy_catalog=False,**kwargs):
   '''
   Generate earthquakes to be used in tomographic inversion. Earthquake locations can 
   be generated randomly within the distance range (deltamin, deltamax), or alternatively
   a ring of earthquakes at a fixed distance from the point (0,0) can be generated.
   In the futures, events may be given as an obspy event catalog.

   args--------------------------------------------------------------------------
   geometry: either 'random' or 'ring'
   
   kwargs------------------------------------------------------------------------
   nevents: number of earthquakes (only use if geometry is 'random')
   deltamin = minimum earthquake distance from (0,0) (only if geometry is 'random')
   deltamax = maximum earthquake distance from (0,0) (only if geometry is 'random')
   ringdist = distance of ring from (0,0). Can be tuple for multiple rings. (only if geometry is 'ring')
   dtheta = spacing between earthquakes in ring, given in degrees. Default = 30.
   '''

   lat0 = kwargs.get('lat0',0.0)
   lon0 = kwargs.get('lon0',0.0)
   depth = kwargs.get('depth',0.0)
   dtheta = kwargs.get('dtheta',30.0)
   nevents = kwargs.get('nevents',0)
   deltamin = kwargs.get('deltamin',0)
   deltamax = kwargs.get('deltamax',0)
   ringdist = kwargs.get('ringdist',50)

   eq_list = []
   n = 1

   if geometry=='random':
      while len(eq_list) < nevents:
         lon = (2.0*deltamax*np.random.random(1)) - deltamax
         lat = (2.0*deltamax*np.random.random(1)) - deltamax
         dist_az = gps2DistAzimuth(lat,lon,lat0,lon0)
         dist_km = dist_az[0]/1000.0
         dist_deg = kilometer2degrees(dist_km)

         if dist_deg >= deltamin and dist_deg <= deltamax:
            eq_list.append((n,lat[0],lon[0],depth))
            n += 1

   elif geometry=='ring':
      theta = np.arange(0,360,dtheta)
      thetarad = np.radians(theta)

      if type(ringdist)==int or type(ringdist)==float:
         lat0 = 0
         lon0 = ringdist
         lats = lon0*np.sin(thetarad)
         lons = lon0*np.cos(thetarad)

         for j in range(0,len(lons)):
            eq_list.append((n,lats[j],lons[j],depth))
            n += 1

      elif type(ringdist==tuple):
         for r in ringdist:
            lat0 = 0
            lon0 = r
            lats = lon0*np.sin(thetarad)
            lons = lon0*np.cos(thetarad)

            for j in range(0,len(lons)):
               eq_list.append((n,lats[j],lons[j],depth))
               n += 1

   elif geometry=='ring2':
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
            eq_list.append((n,lat,lon,depth))
            n += 1

      elif type(ringdist==tuple):
         for r in ringdist:
            d_km = r * ((6371.0*2*np.pi)/360.0)
            for i in range(0,len(theta)):
               bearing = theta[i] 
               destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
               lat = destination[0]
               lon = destination[1]
               eq_list.append((n,lat,lon,depth))
               n += 1

   np.savetxt('earthquake_list',eq_list,fmt=['%d','%5.5f','%5.5f','%5.5f'])
   return eq_list

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

      #read event info
      line = f.readline()
      line = f.readline()
      event_geometry = line.strip().split()[1]
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

      #read station info
      line = f.readline()
      line = f.readline()
      station_geometry = line.strip().split()[1]
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

      #read inversion settings
      line = f.readline()
      line = f.readline()
      dlnVp = line.strip().split()[1]
      line = f.readline()
      dlnVs = line.strip().split()[1]
      line = f.readline()
      dlnQs = line.strip().split()[1]
      line = f.readline()
      sig_est = line.strip().split()[1]
      line = f.readline()
      use_hypo_corr = line.strip().split()[1]
      line = f.readline()
      hypo_corr_km = line.strip().split()[1]
      line = f.readline()
      use_sta_corrTP = line.strip().split()[1]
      line = f.readline()
      sta_corrTP = line.strip().split()[1]
      line = f.readline()
      use_sta_corrAP = line.strip().split()[1]
      line = f.readline()
      sta_corrAP = line.strip().split()[1]
      line = f.readline()
      use_origin_corr = line.strip().split()[1]
      line = f.readline()
      origin_corr = line.strip().split()[1]
      line = f.readline()
      use_evnt_corr = line.strip().split()[1]
      line = f.readline()
      evnt_corr = line.strip().split()[1]
      line = f.readline()
      use_sta_corrTS = line.strip().split()[1]
      line = f.readline()
      sta_corrTS = line.strip().split()[1]
      line = f.readline()
      use_sta_corrAS = line.strip().split()[1]
      line = f.readline()
      sta_corrAS = line.strip().split()[1]
      line = f.readline()
      use_ellip_corr = line.strip().split()[1]
      line = f.readline()
      use_crust_corr = line.strip().split()[1]
      line = f.readline()
      use_elev_corr = line.strip().split()[1]
      line = f.readline()
      use_Qdis_corr = line.strip().split()[1]
      line = f.readline()
      demean = line.strip().split()[1]
      

      #find nevents for 'ring' geometry option
      ringdist = literal_eval(ringdist)
      if event_geometry=='ring':
         if type(ringdist)==int or type(ringdist)==float:
            nevents = 360.0 / literal_eval(dtheta)
         elif type(ringdist)==tuple:
            nevents = len(ringdist) * (360 / literal_eval(dtheta))

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
                    'ringdist':ringdist,
                    'dtheta':float(dtheta),
                    'station_geometry':station_geometry,
                    'nstations':int(nstations),
                    'latmin':float(latmin),
                    'latmax':float(latmax),
                    'lonmin':float(lonmin),
                    'lonmax':float(lonmax),
                    'dlat':float(dlat),
                    'dlon':float(dlon),
                    'dlnVp':literal_eval(dlnVp),
                    'dlnVs':literal_eval(dlnVs),
                    'dlnQs':literal_eval(dlnQs),
                    'sig_est':literal_eval(sig_est),
                    'use_hypo_corr':literal_eval(use_hypo_corr),
                    'hypo_corr_km':literal_eval(hypo_corr_km),
                    'use_sta_corrTP':literal_eval(use_sta_corrTP),
                    'sta_corrTP':literal_eval(sta_corrTP),
                    'use_sta_corrAP':literal_eval(use_sta_corrAP),
                    'sta_corrAP':literal_eval(sta_corrAP),
                    'use_origin_corr':literal_eval(use_origin_corr),
                    'origin_corr':literal_eval(origin_corr),
                    'use_evnt_corr':literal_eval(use_evnt_corr),
                    'evnt_corr':literal_eval(evnt_corr),
                    'use_sta_corrTS':literal_eval(use_sta_corrTS),
                    'sta_corrTS':literal_eval(sta_corrTS),
                    'use_sta_corrAS':literal_eval(use_sta_corrAS),
                    'sta_corrAS':literal_eval(sta_corrAS),
                    'use_ellip_corr':literal_eval(use_ellip_corr),
                    'use_crust_corr':literal_eval(use_crust_corr),
                    'use_elev_corr':literal_eval(use_elev_corr),
                    'use_Qdis_corr':literal_eval(use_Qdis_corr),
                    'demean':literal_eval(demean),
                    't_sig':literal_eval(t_sig),
                    'add_noise':literal_eval(add_noise),
                    'ray_theory':literal_eval(ray_theory)}

      #return run_name,delays_file,phase,nperiods,period_list, \
      #       taup_model,event_geometry,nevents,delta_min,delta_max, \
      #       station_geometry,nstations,latmin,latmax,lonmin,lonmax
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
   station_lats = stations[0,:]
   station_lons = stations[1,:]
   x,y = m(station_lons,station_lats)
   m.scatter(x,y,s=50,marker='^',c='r')

   for event in events:
      lon = event[2]
      lat = event[1]
      x,y = m(lon,lat)
      m.scatter(x,y,s=100,marker='*',c='y')

   plt.savefig('geo_config.pdf',format='pdf')
   #lons2,lats2 = m(lons,lats)
   #m.pcolormesh(lons2,lats2,event_map,vmin=-3.0,vmax=0.1)


def get_event_params(eq_lat,eq_lon):
   dist_az = gps2DistAzimuth(eq_lat,eq_lon,0,0)
   dist_km = dist_az[0]/1000.0
   dist_deg = kilometer2degrees(dist_km)
   az = dist_az[1]
   baz = dist_az[2]
   rotation_angle = -1.0*(az - 90.0)

   return dist_deg,rotation_angle

def get_filter_params(delays_file,phase,Tmin):
   '''
   returns the filter parameters used for cross correlation delay measurements

   args:
   delays_file: h5py delay dataset (path to file)
   phase: 'P' or 'S'
   Tmin: minimum period (string, e.g., '10.0')
   '''
   if os.path.isfile(delays_file):
      f = h5py.File(delays_file)

   data = f[phase]['35'][Tmin]
   filter = data.attrs['filter']
   freqmin = data.attrs['freqmin']
   freqmax = data.attrs['freqmax']
   window = data.attrs['window']

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

   returns-----------------------------------------------------------------------
   omega: frequency axis (rad/s)
   amp: frequency response of filter
   '''
   plot = kwargs.get('plot',False)
   corners = kwargs.get('corners',2)
   nyquist = 0.5 * sampling_rate
   fmin = freqmin/nyquist
   fmax = freqmax/nyquist
   
   if filter_type == 'bandpass':
      b, a = iirfilter(corners, [fmin,fmax], btype='band', ftype='butter')
      freq_range = np.linspace(0,0.15,200)
      w, h = freqz(b,a,worN=freq_range)
      omega = sampling_rate * w
      omega_hz = (sampling_rate * w) / (2*np.pi)
      amp = abs(h)

   if plot:
      fig,axes = plt.subplots(2)
      axes[0].plot(omega,amp)
      axes[0].set_xlabel('frequency (rad/s)')
      axes[0].set_ylabel('amplitude')
      axes[1].plot(omega_hz,amp)
      axes[1].set_xlabel('frequency (Hz)')
      axes[1].set_ylabel('amplitude')

   return omega, amp

def make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,**kwargs):
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

   #-----------------------------------------------------------------------------
   #Step 1) Find earthquake distance and back azimuth (from [0,0])
   #-----------------------------------------------------------------------------
   dist_deg, rotation_angle = get_event_params(eq_lat,eq_lon)

   if dist_deg <= 35.0:
      print dist_deg
      raise ValueError('The earthquake is too close')
   #elif dist_deg >= 80.0:
   elif dist_deg >= 1300.0:
      print dist_deg
      raise ValueError('The earthquake is too far')

   #-----------------------------------------------------------------------------
   #Step 2) Create interpolated delay map for given distance
   #-----------------------------------------------------------------------------
   event_list = ['35','50','65','80'] #distances of modelled events
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

   #get point information

   #xi = np.arange(-30,30.0,0.1) #list of x values to interpolate on
   #yi = np.arange(-30,30.0,0.1) #list of y values to interpolate on
   for points in delay_points:
      x = points[0]
      y = points[1]

      #rotate by back azimuth
      y,x = rotate_delays(y,x,0.0,0.0,rotation_angle)

      delays = points[2]
      dti = mlab.griddata(x,y,delays,lons_i,lats_i,interp='linear')
      dti = np.nan_to_num(dti)
      delay_maps.append(dti.data)

   zi = np.array((35,50,65,80))
   delay_array = np.array(delay_maps) 
   event_interpolator = interpolate.RegularGridInterpolator((zi,lats_i,lons_i),delay_array)

   #interpolate new map for given distance
   xx,yy = np.meshgrid(lons_i,lats_i)
   event_map = event_interpolator((dist_deg,yy,xx)) 

   if plot:
      lons,lats = np.meshgrid(lons_i,lats_i)
      m = Basemap(projection='hammer',lon_0=0,resolution='l')
      m.drawcoastlines()
      m.drawmeridians(np.arange(0,351,10))
      m.drawparallels(np.arange(-80,81,10))
      lons2,lats2 = m(lons,lats)
      m.pcolormesh(lons2,lats2,event_map,vmin=-3.0,vmax=0.1)
      m.colorbar()
      lon_eq,lat_eq = m(eq_lon,eq_lat)
      m.scatter(lon_eq,lat_eq,marker='*',s=100,c='y')
      plt.title('{} delays at {} s'.format(phase,Tmin))

      if not return_axis:
         plt.show()

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
      lats = stations[0,:]
      lons = stations[1,:]

   delay_interpolator = interpolate.RegularGridInterpolator((lons_i,lats_i),event_map)
   station_delays = delay_interpolator((lats,lons))

   if pass_figure_axis:
      lons_bsmp, lats_bsmp = figure_axis(lons,lats)
      figure_axis.scatter(lons_bsmp,lats_bsmp,marker='^',s=50,c=station_delays,vmin=-3.0,vmax=0.1)
      plt.show()

   return station_delays
   

def write_input(eq_lat,eq_lon,eq_dep,ievt,stations,phase,delays_file,Tmin,taup_model,filename,
               raytheory=False,tt_from_raydata=True,**kwargs):
   '''
   write an input file for globalseis finite frequency tomography software.
   each earthquake and datatype (P,S,etc...) has it's own input file

   args--------------------------------------------------------------------------
   eq_lat: earthquake latitude (deg)
   eq_lon: earthquake longitude (deg)
   eq_dep: earthquake depth (km)
   stations: stations
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
   sampling_rate = kwargs.get('sampling_rate',20.0)
   n_bands = kwargs.get('n_bands',1) # spectral bands used (TODO setup more than one)
   kunit = kwargs.get('kunit',1) #unit of noise (1 = nm)
   rms0 = kwargs.get('rms0',0) #don't know what this is
   plot_figure = kwargs.get('plot_figure',False)
   dist_min = kwargs.get('dist_min',30.0)
   dist_max = kwargs.get('dist_max',90.0)
   t_sig = kwargs.get('t_sig',0.0)
   add_noise = kwargs.get('add_noise',False)
   fake_SKS_header = kwargs.get('fake_SKS_header',False)

   #create taup model------------------------------------------------------------
   tt_model = TauPyModel(taup_model)

   #get filter parameters--------------------------------------------------------
   print 'Tmin = ', Tmin
   filter_type, freqmin,freqmax, window = get_filter_params(delays_file,phase,Tmin)
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
   lats_i = np.arange(-30.0,30.0,0.1)
   lons_i = np.arange(-30.0,30.0,0.1)

   if plot_figure:
      event_map,figure_axis = make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,lats_i=lats_i,
                                                   lons_i=lons_i,plot=False,return_axis=False)
   else:
      event_map = make_event_delay_map(eq_lat,eq_lon,phase,delays_file,Tmin,
                                       lats_i=lats_i,lons_i=lons_i,return_axis=False)

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

   station_lats = stations[0,:]
   station_lons = stations[1,:]
   n_stations = len(station_lats)
   station_elevation = 0.0

   for i in range(0,n_stations):
      #get ray theoretical travel time 
      dist_deg, rotation_angle = get_event_params(eq_lat,eq_lon)
      #find event distance
      event_distaz = gps2DistAzimuth(eq_lat,eq_lon,station_lats[i],station_lons[i])
      event_dist_deg = kilometer2degrees((event_distaz[0]/1000.0)) 

      #skip station if too close or too far from source
      if event_dist_deg <= dist_min or event_dist_deg >= dist_max:
         continue

      ray_theory_arr = tt_model.get_travel_times(eq_dep,event_dist_deg,[phase,'p','Pdiff'])
      ray_theory_travel_time = ray_theory_arr[0].time
      delay_time = station_delays[i] 

      #tobs = ray_theory_travel_time + delay_time     #don't add if sign convention is negative for late arrivals
      tobs = ray_theory_travel_time - delay_time
      #tobs = ray_theory_travel_time #THIS IS FOR BACKGROUND TESTING, REMOVE!! : RM082016
      #print 'station lats, station lons ', station_lats[i], station_lons[i]
      #print 'the ray theoretical travel time is ', ray_theory_travel_time
      print 'the delay time is', delay_time
      fdelays.write('{}'.format(delay_time)+'\n')
      #print 'the travel time observation is ', tobs

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
      f.write('{} {} {} {} {} {}'.format(tobs,t_sig,corcoeft,nbt,window_len,'#tobs,tsig,corcoeft,nbt,window')+'\n') 

      #write line 5--------------------------------------------------------------
      f.write('{}'.format(0)+'\n')

def write_run_raydata(run_name,nevents):
   '''
   writes the file run_raydata.py to run program 'raydata'
   '''

   f = open('run_raydata.py','w')
   f.write('import os'+'\n') 
   f.write('import subprocess'+'\n') 
   f.write('import numpy as np'+'\n') 
   f.write('executable = "/geo/home/romaguir/globalseis/bin/raydata2"'+'\n')
   f.write('{} = "{}"'.format('run_name',run_name)+'\n')
   f.write('{} = {}'.format('nevents',nevents)+'\n')
   f.write('for i in range(1,nevents+1):'+'\n')
   f.write('   {}'.format("fname = 'in.rd'"+'\n'))
   f.write('   {}'.format("f = open(fname,'w')"+'\n'))
   f.write('   {}'.format("event_id = '{}_{}'.format(run_name,i)"+'\n'))
   f.write('   {}'.format("line = '{}'.format('/geo/home/romaguir/utils/IASP91')"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("line = '{} {}'.format(1,5)"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("line = '{}'.format(event_id)"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("f.close()"+'\n'))
   f.write('   {}'.format("subprocess.call(executable+' < '+fname,shell=True)"+'\n'))

def write_run_voxelmatrix(run_name,nevents,model_par):
   '''
   writes the file run_voxelmatrix.py to run program 'voxelmatrix'

   args--------------------------------------------------------------------------
   run_name
   nevents
   model_par: model parameter to invert for (Vp, Vs)
   '''
   f = open('run_voxelmatrix.py','w')
   f.write('import os'+'\n') 
   f.write('import subprocess'+'\n') 
   f.write('import numpy as np'+'\n') 
   f.write('executable = "/geo/home/romaguir/globalseis/bin/voxelmatrix"'+'\n')
   f.write('{} = "{}"'.format('run_name',run_name)+'\n')
   f.write('{} = {}'.format('nevents',nevents)+'\n')
   f.write('for i in range(1,nevents+1):'+'\n')
   f.write('   {}'.format("fname = 'in.vxm'"+'\n'))
   f.write('   {}'.format("f = open(fname,'w')"+'\n'))
   f.write('   {}'.format("event_id = '{}_{}'.format(run_name,i)"+'\n'))

   if model_par == 'P':
      f.write('   {}'.format("line = '{} {} {}'.format(1,0,0)"+'\n'))
   elif model_par == 'S':
      f.write('   {}'.format("line = '{} {} {}'.format(0,1,0)"+'\n'))
   elif model_par == 'Qs':
      f.write('   {}'.format("line = '{} {} {}'.format(0,0,1)"+'\n'))
   else:
      print 'model parameter ', model_par, ' unrecognized'

   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("line = '{}'.format(20.0)"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("line = '{}'.format(event_id)"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("line = '{}'.format('Y')"+'\n'))
   f.write('   {}'.format("f.write(line+'\\n')"+'\n'))
   f.write('   {}'.format("f.close()"+'\n'))
   f.write('   {}'.format("subprocess.call(executable+' < '+fname,shell=True)"+'\n'))

def write_run_assemblematrix(param_dict):
   '''
   writes the input file for assemblematrixv (in.asm) and runs the program. 
   in.asm specifies which parameters to invert for, as well as which corrections
   to apply. everything is set in the file 'inparam_tomo'.  To repeat an inversion
   which different parameters, manually modify in.asm and rename the output.  
   '''
   nevents = param_dict['nevents']
   run_name = param_dict['run_name']

   f = open('in.asm','w')
   f.write('{} {} '.format(param_dict['dlnVp'],param_dict['sig_est']))
   f.write('#invert for dlnVp (y/n), estimated parameter sigma (0.01=1% Vp anomalies \n')
   f.write('{} {} '.format(param_dict['dlnVs'],param_dict['sig_est']))
   f.write('#invert for dlnVs (y/n) \n')
   f.write('{} {} '.format(param_dict['dlnQs'],param_dict['sig_est']))
   f.write('#invert for dlnQs (y/n) \n')
   f.write('{} {} '.format(param_dict['use_hypo_corr'],param_dict['hypo_corr_km']))
   f.write('#hypocenter correction in km \n')
   f.write('{} {} '.format(param_dict['use_sta_corrTP'],param_dict['sta_corrTP']))
   f.write('#station correction t-P (s) \n')
   f.write('{} {} '.format(param_dict['use_sta_corrAP'],param_dict['sta_corrAP']))
   f.write('#station correction dlnA-P (dimensionless) \n')
   f.write('{} {} '.format(param_dict['use_origin_corr'],param_dict['origin_corr']))
   f.write('#origin time correction (s) \n')
   f.write('{} {} '.format(param_dict['use_evnt_corr'],param_dict['evnt_corr']))
   f.write('#event correction dlnA-P \n')
   f.write('{} {} '.format(param_dict['use_sta_corrTS'],param_dict['sta_corrTS']))
   f.write('#station correction t-S (s) \n')
   f.write('{} {} '.format(param_dict['use_sta_corrAS'],param_dict['sta_corrAS']))
   f.write('#station correction dlnA-S \n')
   f.write('{} {} {} {} '.format(param_dict['use_ellip_corr'],param_dict['use_crust_corr'],
                                 param_dict['use_elev_corr'],param_dict['use_Qdis_corr']))
   f.write('#corrections: ellipticity, curst, station elevation, Q-dispersion \n')
   f.write('inversion \n')

   for i in range(1,nevents+1):
      matrix_id = run_name+'_{}'.format(i) 
      f.write('matrixT.{}'.format(matrix_id)+'\n') 

   if param_dict['demean'] == 1:
      f.write('demean \n')

   f.write('stop')

def write_inmpisolve(run_name='inversion',**kwargs):
   chi2 = kwargs.get('chi2',0.0)   
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
   f.write('mpirun -np 16 '+executable+' < in.mpisolvetomo > out.mpisolvetomo')

def write_doit(run_inversion=False,tt_from_raydata=True):
   assemblematrix_exe = '/geo/home/romaguir/globalseis/bin/assemblematrixv'
   f = open('doit','w')
   f.write('#!/bin/bash \n')
   f.write('python run_raydata.py \n')
   if tt_from_raydata:
      f.write('/geo/home/romaguir/globalseis/bin/tt_from_raydata \n')
      f.write('python run_raydata.py \n')
      f.write('rm tt_out* \n')
      f.write('rm xcor* \n')
   f.write('python run_voxelmatrix.py \n')
   f.write(assemblematrix_exe+' < in.asm \n')

   if run_inversion:
      f.write('qsub submit_mpisolve.qsub \n')

   subprocess.call('chmod +x doit',shell=True)
