#receiver function class 
#01/06/16
import obspy
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.signal import deconvolve
from obspy.signal import rotate
from obspy.taup.taup import getTravelTimes
from obspy.taup import TauPyModel
from obspy.core.util.geodetics import gps2dist_azimuth
from obspy.core.util.geodetics import kilometer2degrees
from seis_tools.decon import water_level
from seis_tools.decon import damped_lstsq
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d


class receiver_function(object):

   def __init__(self, ses3d_seismogram):

       #inherit information from instance of ses3d_seismogram()
       self.ses3d_seismogram = ses3d_seismogram
       self.eq_depth = ses3d_seismogram.sz/1000.0

       #initialize receiver function specific information
       dist_az        = gps2dist_azimuth((90.0-self.ses3d_seismogram.rx),
                                         self.ses3d_seismogram.ry, 
                                        (90.0-self.ses3d_seismogram.sx),
                                         self.ses3d_seismogram.sy,
                                         a = 6371000.0, f = 0.0)
       self.back_az   = dist_az[1]
       self.az        = dist_az[2]
       self.delta_deg = kilometer2degrees(dist_az[0]/1000.0)
       self.time      = np.zeros(100)
       self.prf       = np.zeros(100)
       self.srf       = np.zeros(100)
       self.r = np.zeros(100)
       self.t = np.zeros(100)
       self.z = np.zeros(100)
       self.slowness = 0.0
       self.pierce_dict = []
       self.window_start = 0.0
       self.window_end   = 0.0

   def get_P_rf(self, window_start=-10.0, window_end=100.0,
                      wl = 0.1,  rotation_method = 'RTZ', 
                      type = 'earth_model',plot = False,
                      decon_type='water_level'):

       self.window_start = window_start
       self.window_end   = window_end

       #initialize receiver function time series
       len_s    = self.window_end - self.window_start
       len_i    = int(len_s/self.ses3d_seismogram.dt)

       if(type == 'earth_model'):

          model = TauPyModel(model='pyrolite_5km')
          tt = model.get_travel_times(source_depth_in_km = self.eq_depth,
                                  distance_in_degree = self.delta_deg,
                                  phase_list=["P","P660s"])

          #just in case there's more than one phase arrival, loop through tt list
          for i in range(0,len(tt)):
             if tt[i].name == 'P':
                p_wave_arrival = tt[i].time
                p_i             = int(p_wave_arrival / self.ses3d_seismogram.dt)
             elif tt[i].name == 'P660s':
                self.slowness = tt[i].ray_param * (np.pi/180.0)

       
       elif(type == 'toy_model'):
          p_i = np.argmax(self.ses3d_seismogram.trace_z)
          p_wave_arrival = p_i * self.ses3d_seismogram.dt

       #window seismograms to [P-window_start : P+window_end] 
       wi_start   = int((p_wave_arrival+window_start)/self.ses3d_seismogram.dt) 
       wi_end     = int((p_wave_arrival+window_end)/self.ses3d_seismogram.dt)  
       trace_x_windowed = self.ses3d_seismogram.trace_x[wi_start:wi_end]
       trace_y_windowed = self.ses3d_seismogram.trace_y[wi_start:wi_end]
       trace_z_windowed = self.ses3d_seismogram.trace_z[wi_start:wi_end]

       #find incidence angle from P wave amplitudes. first rotate to rtz
       r_here, t_here  = rotate.rotate_ne_rt(trace_x_windowed, trace_y_windowed, self.back_az)
       z_here          = trace_z_windowed
       r_amp           = np.amax(r_here)
       z_amp           = np.amax(z_here)
       incidence_angle = np.arctan(r_amp/z_amp) * (180/np.pi)

       #rotate
       if rotation_method == 'RTZ':
          self.r, self.t = rotate.rotate_ne_rt(trace_x_windowed,
                                               trace_y_windowed,
                                               self.back_az)
          self.z         = trace_z_windowed
       elif rotation_method == 'LQT' and type == 'earth_model':
          self.z, self.r, self.t = rotate.rotate_zne_lqt(trace_z_windowed,
                                                         trace_x_windowed,
                                                         trace_y_windowed,
                                                         self.back_az,
                                                         incidence_angle)
       elif rotation_method == 'LQT' and type == 'toy_model':
          raise ValueError('rotation method LQT not implemented for type : toy_model')

       #deconvolve Z (apprx. P wave pulse) from R
       self.time = np.linspace(window_start, window_end, len(self.r))
       if decon_type == 'water_level':
          self.prf  = water_level(self.r,self.z,wl)
       elif decon_type == 'damped_lstsq':
          self.prf  = damped_lstsq(self.r,self.z,damping=0.001)

       #plot the two waveforms being deconvolved
       if plot == True:
          plt.plot(self.time, self.r)
          plt.plot(self.time, self.z)

       if decon_type=='water_level':
          #center of the receiver function on the P arrival
          spike       = np.exp((-1.0*(self.time)**2)/0.1)
          spike_omega = np.fft.fft(spike)
          prf_omega   = np.fft.fft(self.prf)
          prf_shifted = spike_omega*prf_omega
          self.prf    = np.real(np.fft.ifft(prf_shifted))

       #normalize
       self.prf /= self.prf.max()
       if rotation_method=='LQT':
          self.prf *= -1.0

   def find_pierce_coor(self,plot='False'):
      import geopy
      from geopy.distance import VincentyDistance

      '''
      given an instance of the receiver function class this function
      returns latitude and longitude of all receiver side pierce points
      of Pds in a given depth range (the default range is 50 - 800 km)
      NOTE:
      be careful that ses3d typically uses colatitude, while
      this function returns latitude '''

      depth_range = np.arange(50,800,5)        #set range of pierce points

      #geodetic info
      bearing     = self.az
      lon_s = self.ses3d_seismogram.sy
      lat_s = 90.0-self.ses3d_seismogram.sx
      lon_r = self.ses3d_seismogram.ry
      lat_r = 90.0-self.ses3d_seismogram.rx
      origin      = geopy.Point(lat_s, lon_s)

      #find how far away the pierce point is
      model  = TauPyModel(model='pyrolite_5km')

      for i in range(0,len(depth_range)):
         phase = 'P'+str(depth_range[i])+'s'
         pierce = model.get_pierce_points(self.eq_depth,self.delta_deg,phase_list=[phase])
         points = pierce[0].pierce
         for j in range(0,len(points)):
            if points[j]['depth'] == depth_range[i] and points[j]['dist']*(180.0/np.pi) > 25.0:
               prc_dist = points[j]['dist']*(180.0/np.pi)
               d_km = prc_dist * ((2*np.pi*6371.0/360.0))
               destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
               lat = destination[0]
               lon = destination[1]
               value = 0
               row = {'depth':depth_range[i],'dist':prc_dist,'lat':lat,'lon':lon,'value':value}
               self.pierce_dict.append(row)

      if plot=='True':
         m = Basemap(projection='hammer',lon_0=0)
         m.drawmapboundary()
         m.drawcoastlines()
         m.drawgreatcircle(lon_s,lat_s,lon_r,lat_r,linewidth=1,color='b',alpha=0.5)

         for i in range(len(self.pierce_dict)):
            x,y = m(self.pierce_dict[i]['lon'],self.pierce_dict[i]['lat'])
            m.scatter(x,y,5,marker='o',color='r')
         plt.show()

   def migrate(self,plot=False):
      import geopy
      from geopy.distance import VincentyDistance

      '''
      This is a rewritten function that combines the functions find_pierce_coor and
      migrate_1d so that it's more efficient.  Still in testing stages. RM 2/6/16
      '''

      depth_range = np.arange(50,800,5)        #set range of pierce points
      value       = np.zeros((len(depth_range)))

      #geodetic info
      bearing     = self.az
      lon_s = self.ses3d_seismogram.sy
      lat_s = 90.0-self.ses3d_seismogram.sx
      lon_r = self.ses3d_seismogram.ry
      lat_r = 90.0-self.ses3d_seismogram.rx
      origin      = geopy.Point(lat_s, lon_s)

      #find how far away the pierce point is
      model  = TauPyModel(model='pyrolite_5km')

      for i in range(0,len(depth_range)):
         phase = 'P'+str(depth_range[i])+'s'
         pierce = model.get_pierce_points(self.eq_depth,self.delta_deg,phase_list=[phase])
         tt     = model.get_travel_times(self.eq_depth,self.delta_deg,phase_list=['P',phase])

         #in case there's duplicate phase arrivals
         for j in range(0,len(tt)):
            if tt[j].name == 'P':
               p_arr = tt[j].time           
            elif tt[j].name == phase:
               phase_arr = tt[j].time

         #determine value 
         Pds_time = phase_arr - p_arr
         i_start  = int((0.0 - self.window_start)/self.ses3d_seismogram.dt)
         i_t      = int(Pds_time/self.ses3d_seismogram.dt) + i_start
         value[i] = self.prf[i_t]

         points = pierce[0].pierce
         for j in range(0,len(points)):
            if points[j]['depth'] == depth_range[i] and points[j]['dist']*(180.0/np.pi) > 20.0:
               prc_dist = points[j]['dist']*(180.0/np.pi)
               d_km = prc_dist * ((2*np.pi*6371.0/360.0))
               destination = VincentyDistance(kilometers=d_km).destination(origin,bearing)
               lat = destination[0]
               lon = destination[1]
               row = {'depth':depth_range[i],'dist':prc_dist,'lat':lat,'lon':lon,'value':value[i]}
               self.pierce_dict.append(row)

      if plot == True:
         plt.plot(value,depth_range)
         plt.gca().invert_yaxis()
         plt.show()

      return value,depth_range


   def plot_pierce_points(self,depth=410.0,ax='None'):
      '''
      Plots pierce points for a given depth.  If an axes object is supplied
      as an argument it will use it for the plot.  Otherwise, a new axes 
      object is created.
      '''

      if ax == 'None':
         m = Basemap(llcrnrlon=0.0,llcrnrlat=-35.0,urcrnrlon=120.0,urcrnrlat=35.0)
      else:
         m = ax

      m.drawmapboundary()
      m.drawcoastlines()
      found_points = False
     
      for i in range(0,len(self.pierce_dict)):
         if self.pierce_dict[i]['depth'] == depth:
            x,y = m(self.pierce_dict[i]['lon'],self.pierce_dict[i]['lat'])
            found_points = True

      if found_points == True:
         m.scatter(x,y,50,marker='+',color='r')
      else:
         print "no pierce points found for the given depth"

   def plot_P_rf(self):
       print len(self.time), len(self.prf)
       zeros = np.zeros((len(self.prf)))
       where = self.prf >= 0
       plt.plot(self.time, self.prf,'k')
       plt.fill_between(self.time,zeros,self.prf,where,facecolor='k')
       plt.show()
     
   def delay_and_sum(self,pds_depth=660.0):
      ''' 
      shift the time axis of a receiver function trace 
      to correct for the moveout of a given phase.
      '''
      #use a reference delta = 45
      ref_deg          = 45.0
      travel_times     = getTravelTimes(ref_deg, self.ses3d_seismogram.sz/1000.0,
                                       phase_list=['P','P660s'])
      P_minus_Pds_ref  = travel_times[0]['time'] - travel_times[1]['time']

      #find delta slowness here
      travel_times     = getTravelTimes(self.delta_deg, self.ses3d_seismogram.sz/1000.0,
                                       phase_list=['P','P660s'])
      P_minus_Pds_here = travel_times[0]['time'] - travel_times[1]['time']
      time_shift       = P_minus_Pds_ref - P_minus_Pds_here
      index_shift    = int(time_shift/self.ses3d_seismogram.dt)
      shifted_trace  = np.roll(self.prf,-1*index_shift)
      self.prf       = shifted_trace
      print "trace shifted by ",time_shift, " seconds"

   def migrate_1d(self,window_start=-10.0,window_end=100.0,plot='True'):
      '''
      migrate receiver functions to depth using a 1d model
      '''
      eq_depth = self.ses3d_seismogram.sz/1000.0
      dist     = self.delta_deg
      
      #depth interval
      d_start  = 50.0
      d_end    = 800.0
      dz       = 5 

      #taup model
      model = TauPyModel(model='pyrolite_5km')
      depth = np.arange(d_start,d_end,dz)
      #depth = depth[::-1]
      value = np.zeros((len(depth)))

      for i in range(0,len(depth)):
         phase    = 'P'+str(depth[i])+'s' 
         tt       = model.get_travel_times(eq_depth,dist,phase_list=['P',phase])

         for j in range(0,len(tt)):
            if tt[j].name == 'P':
               time_p = tt[j].time
            elif tt[j].name == phase:
               time_phase = tt[j].time

         time = time_phase - time_p 
         i_start  = int((0.0 - window_start)/self.ses3d_seismogram.dt)
         i_t      = int(time/self.ses3d_seismogram.dt) + i_start
         value[i] = -1.0*self.prf[i_t]
         self.pierce_dict[i]['value'] = self.prf[i_t]

      if plot == 'True':
         plt.plot(value,depth)
         plt.gca().invert_yaxis()
         plt.show()

      return value,depth

   def migrate_3d(self,window_start=-10.0,window_end=100.0,perturbation_model='none'):

      '''
      Migrates receiver functions using a model with 3d heterogeneity.
      The only difference between migrate_3d and migrate_1d is that
      when migrate_3d calculates (time = time_phase - time_p), both 
      time_phase and time_p have been calculated by ray tracing through
      a 3d model

      #NOTE: This only currently is intended for stations along the equator.
      '''
      #preliminaries-----------------------------------------------------------
      eq_depth = self.ses3d_seismogram.sz/1000.0
      dist     = self.delta_deg
      d_start  = 50.0
      d_end    = 800.0
      dz       = 5
      depth = np.arange(d_start,d_end,dz)

      #taup model--------------------------------------------------------------
      model = TauPyModel(model='pyrolite_5km')
      value = np.zeros((len(depth)))

      #read velocity perturbation model----------------------------------------
      #    note, the specific shape of the input arrays are taken to
      #    match the output of velocity_conversion.py
      #------------------------------------------------------------------------
      to_radians  = np.pi/180.0
      to_degrees  = 180.0/np.pi
      npts_theta  = 201
      npts_rad    = 288
      theta       = np.linspace(0,10.0*to_radians,npts_theta)
      rad         = np.linspace(3490.0,6371.0,npts_rad)
      model_in  = np.loadtxt(perturbation_model)
      #model_in  = np.loadtxt(perturbation_model,skiprows=1)

      dvs_vector  = model_in[:,6]
      dvp_vector  = model_in[:,5]
      #dvs_vector  = model_in[:,3]
      #dvp_vector  = model_in[:,2]

      dvs_array   = np.zeros((npts_theta,npts_rad))
      dvp_array   = np.zeros((npts_theta,npts_rad))

      #define location of plume axis-------------------------------------------
      #TODO: change this so it's not necessary
      ax_theta = 45.0*to_radians
      
      #assign values from vector to array--------------------------------------
      k = 0
      for i in range(0,npts_rad):
         for j in range(0,npts_theta):
            dvs_array[j,i] = dvs_vector[k]
            dvp_array[j,i] = dvp_vector[k]
            k += 1

      #create interpolators----------------------------------------------------
      #   to call the interpolators, give theta in radians and radius in km
      #   where theta is the arc distance from a point to the plume axis
      #------------------------------------------------------------------------
      dvs_interpolator = interp2d(rad,theta,dvs_array,fill_value=0.0)
      dvp_interpolator = interp2d(rad,theta,dvp_array,fill_value=0.0)

      #calculate P travel time (only need to do once)--------------------------
      p_path   = model.get_ray_paths(eq_depth,dist,phase_list=['P'])
      p_path   = p_path[0]
      p_ref    = p_path.time
      p_ray_dist    = p_path.path['dist']
      p_ray_depth   = p_path.path['depth']
      p_ray_time    = p_path.path['time']

      #interpolate p path------------------------------------------------------
      i_len = 2000
      ray_dist_new = np.linspace(0,p_ray_dist[(len(p_ray_dist)-1)],i_len)
      #time
      f = interp1d(p_ray_dist, p_ray_time)
      p_ray_time = f(ray_dist_new)
      #depth
      f = interp1d(p_ray_dist, p_ray_depth)
      p_ray_depth = f(ray_dist_new)
      p_ray_dist = ray_dist_new

      p_time_total = 0.0
      path_len_total = 0.0

      #integrate along P path to determine time--------------------------------
      for i in range(0,len(p_ray_dist)-1):
         r_1 = 6371.0-p_ray_depth[i]
         r_2 = 6371.0-p_ray_depth[i+1]
         theta_1 = p_ray_dist[i]
         theta_2 = p_ray_dist[i+1]
         ray_seg_len = np.sqrt(r_1**2 + r_2**2 - 2*r_1*r_2*np.cos(theta_2-theta_1))
         time_seg    = p_ray_time[i+1] - p_ray_time[i]
         ray_seg_vel = ray_seg_len / time_seg

         #get heterogeneity for the current ray segment
         arc_dist1 = np.abs(ax_theta - theta_1)
         arc_dist2 = np.abs(ax_theta - theta_2)
         dv_1 = dvp_interpolator(r_1,arc_dist1)
         dv_2 = dvp_interpolator(r_2,arc_dist2)
         dv_here = (dv_1+dv_2)/2.0

         #add time-------------------------------------------------------------
         #if using absolute velocity perturbations
         time_here = ray_seg_len / (ray_seg_vel + dv_here)

         #if using percente velocity perturbations
         plume_velocity = ray_seg_vel + ray_seg_vel*(dv_here/100.0)
         time_here = ray_seg_len / plume_velocity

         p_time_total += time_here
         path_len_total += ray_seg_len

      #calculate each pds time-------------------------------------------------
      for i in range(0,len(depth)):

         #get path data
         phase    = 'P'+str(depth[i])+'s'
         pds_path = model.get_ray_paths(eq_depth,dist,phase_list=[phase])
         pds_path = pds_path[0]

         pds_ray_dist  = pds_path.path['dist']
         pds_ray_depth = pds_path.path['depth']
         pds_ray_time  = pds_path.path['time']

         #interpolate pds path----------------------------------------------------
         i_len = 500
         ray_dist_new = np.linspace(0,pds_ray_dist[(len(pds_ray_dist)-1)],i_len)
         #time
         f = interp1d(pds_ray_dist, pds_ray_time)
         pds_ray_time = f(ray_dist_new)
         #depth
         f = interp1d(pds_ray_dist, pds_ray_depth)
         pds_ray_depth = f(ray_dist_new)
         pds_ray_dist = ray_dist_new

         pds_time_total = 0.0

         for j in range(0,len(pds_ray_dist)-1):
            r_1 = 6371.0-pds_ray_depth[j]
            r_2 = 6371.0-pds_ray_depth[j+1]
            theta_1 = pds_ray_dist[j]
            theta_2 = pds_ray_dist[j+1]
            time_seg    = pds_ray_time[j+1] - pds_ray_time[j]
            ray_seg_len = np.sqrt(r_1**2 + r_2**2 - 2*r_1*r_2*np.cos(theta_2-theta_1))
            ray_seg_vel = ray_seg_len / time_seg


            #get heterogeneity for the current ray segment
            arc_dist1 = np.abs(ax_theta - theta_1)
            arc_dist2 = np.abs(ax_theta - theta_2)

            #determine branch
            if ray_seg_vel < 8.0: 
               dv_1 = dvs_interpolator(r_1,arc_dist1)
               dv_2 = dvs_interpolator(r_2,arc_dist2)
               dv_here = (dv_1+dv_2)/2.0
            elif ray_seg_vel >= 8.0: 
               dv_1 = dvp_interpolator(r_1,arc_dist1)
               dv_2 = dvp_interpolator(r_2,arc_dist2)
               dv_here = (dv_1+dv_2)/2.0

            #add time----------------------------------------------------------
            #if using absolute velocity perturbations
            time_here = ray_seg_len / (ray_seg_vel + dv_here)

            #if using percente velocity perturbations
            #plume_velocity = ray_seg_vel + ray_seg_vel*(dv_here/100.0)
            #time_here = ray_seg_len / plume_velocity

            pds_time_total += time_here
            path_len_total += ray_seg_len

         delay = pds_time_total-p_time_total
         #print "P time, Pds time, delay: ",phase,p_time_total,pds_time_total,delay
         #print "Predicted arrival times and delay :", p_path.time, pds_path.time, (pds_path.time-p_path.time)
         pds_delay = pds_time_total - pds_path.time
         print phase, pds_delay
         

         #map value at t = (p_time - pds_time) to depth------------------------
         i_start  = int((0.0 - window_start)/self.ses3d_seismogram.dt)
         i_t      = int(delay/self.ses3d_seismogram.dt) + i_start
         print i_t
         self.pierce_dict[i]['value'] = self.prf[i_t]
      

   def normalize(self):
   
      min = np.amin(self.prf)
      max = np.amax(self.prf)
      sf  = np.max(np.abs(min),np.abs(max))
      self.prf /= (1.0/sf)
