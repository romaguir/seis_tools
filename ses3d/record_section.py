import numpy as np

class ses3d_record_section(object):

   def __init__(self,recfile_directory='../INPUT/recfile_1'):
      recfile_1 = open(recfile_directory,'r')
      self.n_recs = int(recfile_1.readline())
      self.recs = []

      for i in range(0,2*self.n_recs):

         if np.mod(i,2) == 0 :
            line        = recfile_1.readline().strip()
            self.recs.append(line)

         else:
            recfile_1.readline()

      print self.recs

   def plot_all(self,directory,sf=1e5,component='x'):
      import seismograms as s   
      import matplotlib.pylab as plt
      from obspy.core.util.geodetics import gps2DistAzimuth
      from obspy.core.util.geodetics import kilometer2degrees


      for i in range(0,self.n_recs):
         a = s.ses3d_seismogram()
         a.read(directory,self.recs[i])

         #determine epicentral distance
         dist_az   = gps2DistAzimuth((90.0-a.rx),a.ry,(90.0-a.sx),a.sy)
         back_az   = dist_az[1]
         delta_deg = kilometer2degrees(dist_az[0]/1000.0)

         if (component == 'x'):
            plt.plot(a.trace_x*sf + delta_deg, a.t,'k')
         elif (component == 'y'):
            plt.plot(a.trace_y*sf + delta_deg, a.t,'k')
         elif (component == 'z'):
            plt.plot(a.trace_z*sf + delta_deg, a.t,'k')

      plt.show()

   def receiver_functions(self,directory,nf=1,type='earth_model',migrate='False',equator='True'):
      import seismograms as s
      import receiver_functions as rf
      import matplotlib.pylab as plt
      from obspy.taup.taup import getTravelTimes
      from obspy.core.util.geodetics import gps2DistAzimuth
      from obspy.core.util.geodetics import kilometer2degrees
     
      rec_lon  = np.zeros(self.n_recs)
      for i in range(0,self.n_recs):
         a = s.ses3d_seismogram()
         a.read(directory,self.recs[i],integrate=True)
         b = rf.receiver_function(a)

         #TODO for plotting fills
         #where = [False]*(len(b.time))

         if type == 'toy_model' :
            prf = b.get_P_rf(-10.0,60.0,0.1,type='toy_model')
         elif type == 'earth_model' :
            prf = b.get_P_rf(-100.0,100.0,0.1,rotation_method='LQT',type='earth_model',decon_type='damped_lstsq')

         rec_lon[i]  = b.ses3d_seismogram.ry

         #normalize max amplitude to equal nf
         scale = nf/np.amax(b.prf)
         #align on P410s arrival (i.e., moveout correction)
         ref_delta      = 45
         #ref_slowness   = 7.7595   #slowness for P wave at 45 degrees
         #ref_d_slowness = 0.1042   #P - P410s slowness at 45 degrees
         ref_slowness    = 7.7595 
         ref_d_slowness  = 0.2088
         tt = getTravelTimes(delta=b.delta_deg,depth=b.ses3d_seismogram.sz/1000.0,
                             model='ak135',phase_list=['P','P660s'])
         slowness_p       = tt[0]['dT/dD']
         slowness_p410s   = tt[1]['dT/dD']
         delta_slowness   = slowness_p - slowness_p410s
         d_delta_slowness = ref_d_slowness - delta_slowness
         time_shift       = b.delta_deg * d_delta_slowness
         index_shift      = int(time_shift/b.ses3d_seismogram.dt)
         #print "delta_slowness = ", delta_slowness
         #print "time_shift, index_shift = ", time_shift, index_shift

         if migrate == 'False':
            #If you want it aligned on P660s arrivval:
            b.prf = np.roll(b.prf,index_shift)

         if migrate =='True':
            value,dep_m = b.migrate_1d()
            plt.plot(value*scale+b.ses3d_seismogram.ry,dep_m,'k')
        
         #plot
         rf_scaled = (b.prf*scale)+(b.ses3d_seismogram.ry)
         where     = rf_scaled > b.ses3d_seismogram.ry
         plt.figure(1)
         plt.plot(rf_scaled, b.time,'k')
         plt.fill_betweenx(b.time,b.ses3d_seismogram.ry,rf_scaled,where,color='k')

      plt.gca().invert_yaxis()
      plt.xlabel('distance (degrees)')
      plt.ylabel('time after P(s)')
      plt.ylim([0,100])
      plt.gca().invert_yaxis()
      plt.show()
