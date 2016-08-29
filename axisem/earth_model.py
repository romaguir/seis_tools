import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

class earth_model(object):

  def __init__(self):

      #geometry
      self.mantle_thickness = 2891.0
      self.theta_max        = 20.0
      self.npts_theta       = 300
      self.npts_radius      = 750
      self.theta            = np.linspace(0,self.theta_max,self.npts_theta)
      self.radius           = np.linspace((6371.0-self.mantle_thickness),6371.0,self.npts_radius)

      #constants
      self.dvp              = 10.0
      self.dvs              = 5.0
      self.drho             = 5.0

      #1d background model
      self.bg_vp            = np.zeros(self.npts_radius)
      self.bg_vs            = np.zeros(self.npts_radius)
      self.bg_rho           = np.zeros(self.npts_radius)

      #arrays containing model
      self.vp_array         = np.zeros((self.npts_radius,self.npts_theta))
      self.vs_array         = np.zeros((self.npts_radius,self.npts_theta))
      self.rho_array        = np.zeros((self.npts_radius,self.npts_theta))

      #arrays containing model perturbations
      self.dvp_array        = np.zeros((self.npts_radius,self.npts_theta))
      self.dvs_array        = np.zeros((self.npts_radius,self.npts_theta))
      self.drho_array       = np.zeros((self.npts_radius,self.npts_theta))

  def read_background_model(self,path_to_model='./'):
      '''
      Read a background model that has been generated as output from axisem.
      To ensure that axisem generates the background model, check to see
      if 'WRITE_1DMODEL' is set to true in inparam_mesh. The background model
      is linearly interpolated to be the length of self.npts_radius
      '''
      
      #read the data
      data   = np.loadtxt(path_to_model)
      radius = data[:,0]/1000.0
      rho    = data[:,1]
      vp     = data[:,2]
      vs     = data[:,3]

      #interpolate 
      r_new          = self.radius
      f              = interp1d(radius, rho, kind='linear')
      self.rho_bg    = f(r_new)
      f              = interp1d(radius, vp, kind='linear')
      self.vp_bg     = f(r_new)
      f              = interp1d(radius, vs, kind='linear')
      self.vs_bg     = f(r_new)

      #build array
      for i in range(0,self.npts_radius):
          self.rho_array[i,:] = self.rho_bg[i]
          self.vp_array[i,:]  = self.vp_bg[i]
          self.vs_array[i,:]  = self.vp_bg[i]

  def cylinder(self,cyl_rad=200.0,dvp=0.0,dvs=0.0,drho=0.0,type='specfem'):

      '''
      Make a cylindrical anomaly with a constant perturbation
      to wavespeed and density
      '''

      #min_depth = 0.0
      #max_depth = 2891.0
      min_radius = 6371.0-2891.0
      max_radius = 6371.0-100.0

      if(type=='specfem'):
         
          drho_abs = 30.0                      #constant perturbation of 30 kg/m3

          for i in range(0, self.npts_radius):
             km_per_deg    = 2*np.pi*self.radius[i] / 360.0
             cyl_rad_theta = cyl_rad / km_per_deg
             dvp_percent   = -1.0*((0.55/0.30)*(drho_abs/self.rho_bg[i])*100.0)
             dvs_percent   = -1.0*((1.00/0.30)*(drho_abs/self.rho_bg[i])*100.0)
             print dvp_percent
             for j in range(0, self.npts_theta):
                 if (self.theta[j] <= cyl_rad_theta and self.radius[i] <=
                     max_radius and self.radius[i] >= min_radius):
                     #percent perturbations to background
                     self.drho_array[i,j] = -1.0*((drho_abs)/(self.rho_bg[i]))*100.0
                     self.dvp_array[i,j] = dvp_percent
                     self.dvs_array[i,j] = dvs_percent
                     #absolute values
                     self.rho_array[i,j] = self.rho_array[i,j] + drho_abs
                     self.vp_array[i,j]  = self.vp_array[i,j] + (self.vp_array[i,j]*(dvp_percent/100.0))
                     self.vs_array[i,j]  = self.vs_array[i,j] + (self.vs_array[i,j]*(dvs_percent/100.0))
         
      elif(type=='const'):
          for i in range(0, self.npts_radius):
             km_per_deg    = 2*np.pi*self.radius[i] / 360.0
             cyl_rad_theta = cyl_rad / km_per_deg
             for j in range(0, self.npts_theta):
                 if (self.theta[j] <= cyl_rad_theta):
                     self.dvp_array[i,j]  = self.dvp
                     self.dvs_array[i,j]  = self.dvs
                     self.drho_array[i,j] = self.drho

  def plot_1d_model(self):
      plt.subplot(131)
      plt.plot(self.rho_bg,self.radius)
      plt.xlabel('density (kg/m3)')
      plt.ylabel('radius (km)')
      plt.subplot(132)
      plt.plot(self.vp_bg,self.radius)
      plt.xlabel('Vp (km/s)')
      plt.ylabel('radius (km)')
      plt.subplot(133)
      plt.plot(self.vs_bg,self.radius)
      plt.xlabel('Vs (km/s)')
      plt.ylabel('radius (km)')
      plt.show()

  def plot_earth_model(self,type='perturbation'):
      if(type=='model'):
         plt.pcolor(self.theta, self.radius, self.vs_array)
         plt.colorbar()
         plt.show()
      elif(type=='perturbation'):
         plt.pcolor(self.theta, self.radius, self.dvs_array)
         plt.colorbar()
         plt.show()

  def write_1d_model(self,ftype='nd'):
      '''
      write the backround model (vp,vs,rho) to a file.
      ftype options (not yet implemented): 'nd' (named discontinuity), tvel
      '''
      for i in range(0,self.npts_radius):
          print self.bg_rho[i]

  def write_sph(self, filename='hetero.sph'):

      '''
      Write a .sph heterogenity file for use in axisem.
      Format : r(km), theta (degrees), dvp (%), dvs (%), drho (%)
      '''
      f_out = open(filename, 'w')

      count = 0
      for i in range(0, self.npts_radius):
          for j in range(0, self.npts_theta):
              line = (str(self.radius[i])+'  '+str(self.theta[j]) +'  '+
                      str(self.dvp_array[i,j])+'  '+str(self.dvs_array[i,j])+
                      '  '+str(self.drho_array[i,j]))
              f_out.write(line+'\n')

