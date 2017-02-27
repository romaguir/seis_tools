import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from obspy.core.util.geodetics import kilometer2degrees
from seis_tools.models.models_3d import model_3d
from seis_tools.models import models_1d
from scipy.interpolate import interp1d

class velocity_model(object):

   def __init__(self):

      #lookup table parameters
      self.lookup_table = 'Na'
      self.vp_table  = np.zeros((1,1))
      self.vs_table  = np.zeros((1,1))
      self.rho_table = np.zeros((1,1))
      self.T_table_axis = np.zeros(1)
      self.P_table_axis = np.zeros(1)

      #absolute model values (km/s, g/cm^3)
      self.vp_array     = np.zeros((1,1))
      self.vs_array     = np.zeros((1,1))
      self.rho_array    = np.zeros((1,1))
      #absolute model perturbations (km/s, g/cm^3)
      self.dvp_abs      = np.zeros((1,1))
      self.dvs_abs      = np.zeros((1,1))
      self.drho_abs     = np.zeros((1,1))
      #relative model perturbations (%)
      self.dvp_rel      = np.zeros((1,1))
      self.dvs_rel      = np.zeros((1,1))
      self.drho_rel     = np.zeros((1,1))

      #temperature (K)
      self.T = np.zeros((1,1))
      self.T_adiabat    = np.zeros(1)
      self.T_plume_axis = np.zeros(1)
      self.pot_T = 0
      #delta T relative to adiabat (K)
      self.delta_T      = np.zeros((1,1))

      #
      self.P1d   = np.zeros(1)
      self.vp1d  = np.zeros(1)
      self.vs1d  = np.zeros(1)
      self.rho1d = np.zeros(1)
      self.npts_theta  = 201
      self.npts_rad    = 288
      self.theta       = np.linspace(0,10.0,self.npts_theta)
      self.dpr         = 3.472222e-3
      self.dpth        = 8.72664677444e-4
      self.rad_km      = np.linspace(3493,6371,self.npts_rad)
      self.dpr_km      = self.rad_km[1]-self.rad_km[0]

      #
      #self.coor_x = np.zeros(


   def set_adiabat(self,adiabat_file='none',**kwargs):
      '''
      set reference temperature profile from an external file
      Pyrolite adiabats calculated for Stx 11 are located in ~/utils/PyroliteAdiabats

      '''
      ad = np.loadtxt(adiabat_file)
      self.T_adiabat = ad[:,0]
      self.P1d = ad[:,1]
      self.pot_T = self.T_adiabat[0]

   def read_T_snapshot(self,snapshot='none',give_adiabat=False,adiabat_file='none'):
      '''
      Reads in temperature snapshot that has been generated from sepran postprocessing.
   
      Params:
      snapshot : the path to the file
      '''
      T_file = np.loadtxt(snapshot)
      T_vec  = T_file[:,2]        #in sepran output, T is 3d column
      T_vec += 273.15             #sepran output is in C, tables use K
      self.dpth       = 8.72664677444e-4
      self.dpr        = 3.472222e-3
      self.dpr_dim    = 10.0278

      self.T = np.zeros((self.npts_rad,self.npts_theta))
      self.delta_T = np.zeros((self.npts_rad,self.npts_theta))

      k = 0
      for i in range(0,self.npts_rad):
         for j in range(0,self.npts_theta):
            self.T[(self.npts_rad-1)-i,j] = T_vec[k]
            k+=1

      self.T_adiabat    = self.T[:,(self.npts_theta-1)]
      self.T_plume_axis = self.T[:,0]

      if give_adiabat:
         ad_file = np.loadtxt(adiabat_file)
         ad_vec = ad_file[:,2]
         ad_vec += 273.15
         ad_T = np.zeros((self.npts_rad,self.npts_theta)) 
         k = 0
         for i in range(0,self.npts_rad):
            for j in range(0,self.npts_theta):
               ad_T[(self.npts_rad-1)-i,j] = ad_vec[k]
               k+=1

         self.T_adiabat    = ad_T[:,(self.npts_theta-1)]

      #get delta T
      for i in range(0,self.npts_rad):
         for j in range(0,self.npts_theta):
            self.delta_T[i,j] = self.T[i,j] - self.T_adiabat[i]

   def read_lookup_table(self,composition='pyrolite',lookup_table='none',**kwargs):
      '''
      Reads a Perplex generated lookup table in the *.tab format
      Default is a pyrolite composition from Stixrude and Bertelloni, 2011

      Params
      composition:chemical model
                  current options: 'pyrolite', 'morb', 'harzburgite', 'mix'
                  if using 'mix', must also specify kwarg 'basalt_fraction' as value (0,1)
                  

      lookup_table : if chemical model = 'other', provide the path to the lookup table

      **kwargs
      hdf5: If set to True, reads data from an h5py file (faster). currently compositional mixtures 
            will only work if this is set to True.
      basalt_fraction: fraction of basalt in compositional mixture (0 - 1)
      '''
      hdf5 = kwargs.get('hdf5',False)
      table_directory = '/geo/home/romaguir/utils/lookup_tables/'
      h5_loc = table_directory+'lookup_tables.hdf5'
      basalt_fraction = kwargs.get('basalt_fraction','NA')
      database = kwargs.get('database','stx11')

      if hdf5 == False:
         if composition == 'pyrolite':
            if database == 'stx11':
               lookup_table = table_directory+'stx11_pyr_WM_Q7_20s_aboveTmelt.tab'
         elif composition == 'morb':
            if database == 'stx11':
               lookup_table = table_directory+'stx11_morb_WM_Q7_20s_aboveTmelt.tab'
            elif database == 'sfo08':
               lookup_table = table_directory+'MORB_sfo08.tab'
         elif composition == 'harzburgite':
            if database == 'stx11':
               lookup_table = table_directory+'stx11_harz_WM_Q7_20s_aboveTmelt.tab'
            elif database == 'sfo08':
               lookup_table = table_directory+'Harzburgite_sfo08.tab'

         self.lookup_table = lookup_table

         #Read table information------------------------------------------------------------
         info = open(lookup_table,'r')
         junk = info.readline().strip()
         junk = info.readline().strip()
         junk = info.readline().strip()
         junk = info.readline().strip()
         P0 = info.readline().strip()
         dP = info.readline().strip()
         nP = info.readline().strip()
         junk = info.readline().strip()
         T0 = info.readline().strip()
         dT = info.readline().strip()
         nT = info.readline().strip()
         info.close()
         nP=int(nP)
         nT=int(nT)
         dP=float(dP)
         dT=float(dT)
         P0=float(P0)
         T0=float(T0)

         v_file  = np.loadtxt(lookup_table,skiprows=13)
         rho_vec = v_file[:,4]
         vp_vec  = v_file[:,8]
         vs_vec  = v_file[:,9]

         self.vp_table  = np.zeros((nT,nP))
         self.vs_table  = np.zeros((nT,nP))
         self.rho_table = np.zeros((nT,nP))

         k = 0
         for i in range(0,nT):
             for j in range(0,nP):
                 self.vp_table[i,j]  = vp_vec[k]
                 self.vs_table[i,j]  = vs_vec[k]
                 self.rho_table[i,j] = rho_vec[k]
                 k+=1

      elif hdf5 == True:
         if composition != 'mix':
            h5file = h5py.File(h5_loc,'r')
            self.vp_table = h5file[composition]['vp'][:]
            self.vs_table = h5file[composition]['vs'][:]
            self.rho_table = h5file[composition]['rho'][:]
            P0 = h5file[composition]['P0'].value
            T0 = h5file[composition]['T0'].value
            nP = h5file[composition]['nP'].value
            nT = h5file[composition]['nT'].value
            dP = h5file[composition]['dP'].value
            dT = h5file[composition]['dT'].value
         else:
            h5file = h5py.File(h5_loc,'r')
            morb_vp = h5file['morb']['vp'][:]
            morb_vs = h5file['morb']['vs'][:]
            morb_rho = h5file['morb']['rho'][:]
            harz_vp = h5file['harzburgite']['vp'][:]
            harz_vs = h5file['harzburgite']['vs'][:]
            harz_rho = h5file['harzburgite']['rho'][:]
            self.vp_table = basalt_fraction * morb_vp + (1-basalt_fraction)*harz_vp
            self.vs_table = basalt_fraction * morb_vs + (1-basalt_fraction)*harz_vs
            self.rho_table = basalt_fraction * morb_rho + (1-basalt_fraction)*harz_rho
            P0 = h5file['morb']['P0'].value
            T0 = h5file['morb']['T0'].value
            nP = h5file['morb']['nP'].value
            nT = h5file['morb']['nT'].value
            dP = h5file['morb']['dP'].value
            dT = h5file['morb']['dT'].value
            

      self.P_table_axis = np.arange(P0,((nP-1)*dP+P0+dP),dP)
      self.T_table_axis = np.arange(T0,((nT-1)*dT+T0+dT),dT)

   def velocity_conversion(self,two_dimensional=True):
      from scipy.interpolate import interp2d

      #create interpolant
      vp_interpolator  = interp2d(self.P_table_axis, self.T_table_axis, self.vp_table)
      vs_interpolator  = interp2d(self.P_table_axis, self.T_table_axis, self.vs_table)
      rho_interpolator = interp2d(self.P_table_axis, self.T_table_axis, self.rho_table)

      def get_vals_1d(self,gravity='constant'):
         '''
         get 1d profiles of pressure, Vp, Vs, and density
         '''
         self.P1d   = np.zeros((self.npts_rad))
         self.vp1d  = np.zeros((self.npts_rad))
         self.vs1d  = np.zeros((self.npts_rad))
         self.rho1d = np.zeros((self.npts_rad))

         if gravity == 'constant':
            g = 9.8
         else:
            print "only constant gravity is currently implemented"

         for i in range(0,self.npts_rad):
           if self.T_adiabat[i] < 300:
              T_here = 300.0
           else:
              T_here = self.T_adiabat[i]
           
           self.vp1d[i]    = vp_interpolator(self.P1d[i],T_here)
           self.vs1d[i]    = vs_interpolator(self.P1d[i],T_here)
           self.rho1d[i]   = rho_interpolator(self.P1d[i],T_here)
           weight_of_layer = self.rho1d[i]*g*self.dpr_km*1000.0
           weight_of_layer *= 1e-5         #convert form Pa to bar

           if i+1 < self.npts_rad:
              self.P1d[i+1] = self.P1d[i] + weight_of_layer


      def get_vals_2d(self):

         #initialize arrays
         self.vp_array  = np.zeros((self.npts_rad,self.npts_theta))
         self.vs_array  = np.zeros((self.npts_rad,self.npts_theta))
         self.rho_array = np.zeros((self.npts_rad,self.npts_theta))

         self.dvp_abs  = np.zeros((self.npts_rad,self.npts_theta))
         self.dvs_abs  = np.zeros((self.npts_rad,self.npts_theta))
         self.drho_abs = np.zeros((self.npts_rad,self.npts_theta))

         self.dvp_rel  = np.zeros((self.npts_rad,self.npts_theta))
         self.dvs_rel  = np.zeros((self.npts_rad,self.npts_theta))
         self.drho_rel = np.zeros((self.npts_rad,self.npts_theta))

         for i in range(0,self.npts_rad):
            for j in range(0,self.npts_theta):
               #absolute values
               self.vp_array[i,j]  = vp_interpolator(self.P1d[i],self.T[i,j])
               self.vs_array[i,j]  = vs_interpolator(self.P1d[i],self.T[i,j])
               self.rho_array[i,j] = rho_interpolator(self.P1d[i],self.T[i,j])
               #absolute perturbations
               self.dvp_abs[i,j]   = self.vp_array[i,j] - self.vp1d[i]
               self.dvs_abs[i,j]   = self.vs_array[i,j] - self.vs1d[i]
               self.drho_abs[i,j]  = self.rho_array[i,j] - self.rho1d[i]
               self.drho_abs[i,j] /= 1000.0 #convert from kg/m3 to g/cm3
               #percent perturbations
               self.dvp_rel[i,j]   = ((self.vp_array[i,j] - self.vp1d[i])/self.vp1d[i])*100.0
               self.dvs_rel[i,j]   = ((self.vs_array[i,j] - self.vs1d[i])/self.vs1d[i])*100.0
               self.drho_rel[i,j]  = ((self.rho_array[i,j] - self.rho1d[i])/self.rho1d[i])*100.0
              

      def main(self,two_dimensional):
         get_vals_1d(self,gravity='constant')

         if two_dimensional == True:
            get_vals_2d(self)

      main(self,two_dimensional)

   def cylinder(self,deltaT=300.0,radius=200.0,start_depth=100.0):
      '''
      creates a cylindrical temperature anomaly with a gaussian blur
      
      usage: First read a temperature snapshot.

      Params:
      deltaT : cylinder excess temperature
      radius : cylinder radius (im km)
      start_depth: starting depth of the cylinder (default = 100.0)
      '''
      T_ref = self.T_adiabat[::-1]
      T_here = np.zeros((self.npts_rad,self.npts_theta))

      for i in range(0,self.npts_rad):
         print T_ref[i]
         km_per_degree = self.rad_km[i]*2*np.pi/360.0 

         for j in range(0,self.npts_theta):

            cyl_th_here  = radius / km_per_degree
            th_here      = self.theta[j]
            depth_here   = 6371.0 - self.rad_km[i]
            if th_here <= cyl_th_here and depth_here > 100.0:
               T_here[(self.npts_rad-1)-i,j] = T_ref[i] + deltaT
            else:
               T_here[(self.npts_rad-1)-i,j] = T_ref[i]

      filtered = gaussian_filter(T_here,sigma=[0,3])
      self.T = filtered
      #self.T = T_here

   def constant_cylinder(self,radius,dvp=-5.0,dvs=-5.0,drho=-1.0,start_depth=100.0,end_depth=2800.0,**kwargs):
      '''
      generates a cylindrical anomaly with constant perturbation in Vp, Vs, and density
      use this *after* running 'velocity_conversion'
      args-----------------------------------------------------------------------
      #   radius: radius of cylinder in km
      #   dvp: vp perturbation in %
      #   dvs: vs perturbation in %
      #   rho: density perturbation in %
      #   start_depth: starting depth of cylinder

      #kwargs--------------------------------------------------------------------
      sigma_blur: sigma for gaussian smoothing (default = 3.0)
      '''
      self.dvp_rel  = np.zeros((self.npts_rad,self.npts_theta))
      self.dvs_rel  = np.zeros((self.npts_rad,self.npts_theta))
      self.drho_rel = np.zeros((self.npts_rad,self.npts_theta))
      self.dvp_abs  = np.zeros((self.npts_rad,self.npts_theta))
      self.dvs_abs  = np.zeros((self.npts_rad,self.npts_theta))
      self.drho_abs = np.zeros((self.npts_rad,self.npts_theta))
      sigma_blur = kwargs.get('sigma_blur',3.0)
      sigma_blur_vert = kwargs.get('sigma_blur_vert',0.5)

      #T_ref = self.T_adiabat[::-1]
      #T_here = np.zeros((self.npts_rad,self.npts_theta))

      for i in range(0,self.npts_rad):
         km_per_degree = self.rad_km[i]*2*np.pi/360.0 

         #get 1d values
         vp_here = self.vp1d[::-1][i]
         vs_here = self.vs1d[::-1][i]
         rho_here = self.rho1d[::-1][i]

         #calculate absolute values of perturbations
         dvp_abs = vp_here * (dvp/100.0)
         dvs_abs = vs_here * (dvs/100.0)
         drho_abs = rho_here * (drho/100000.0)

         for j in range(0,self.npts_theta):

            cyl_th_here  = radius / km_per_degree
            th_here      = self.theta[j]
            depth_here   = 6371.0 - self.rad_km[i]

            if th_here <= cyl_th_here and depth_here > start_depth and depth_here < end_depth:
               self.dvp_abs[(self.npts_rad-1)-i,j] += dvp_abs
               self.dvs_abs[(self.npts_rad-1)-i,j] += dvs_abs
               self.drho_abs[(self.npts_rad-1)-i,j] += drho_abs
               self.dvp_rel[(self.npts_rad-1)-i,j] += dvp
               self.dvs_rel[(self.npts_rad-1)-i,j] += dvs
               self.drho_rel[(self.npts_rad-1)-i,j] += drho

      self.dvp_abs = gaussian_filter(self.dvp_abs,sigma=[sigma_blur_vert,sigma_blur])
      self.dvs_abs = gaussian_filter(self.dvs_abs,sigma=[sigma_blur_vert,sigma_blur])
      self.drho_abs = gaussian_filter(self.drho_abs,sigma=[sigma_blur_vert,sigma_blur])
      self.dvp_rel = gaussian_filter(self.dvp_rel,sigma=[sigma_blur_vert,sigma_blur])
      self.dvs_rel = gaussian_filter(self.dvs_rel,sigma=[sigma_blur_vert,sigma_blur])
      self.drho_rel = gaussian_filter(self.drho_rel,sigma=[sigma_blur_vert,sigma_blur])

   def plot_field(self,type='abs',contour=True,**kwargs):
      '''
      imshow plots the vp, vs, and density fields.
    
      params:
      type - can be one of I) 'abs'      (absolute values of fields)
                          II) 'abs_diff' (absolute values of perturbations)
                         III) 'rel_diff' (percent perturbations)
      kwargs:
      savefig - save figure 
      figname - name of figure
      '''
      savefig = kwargs.get('savefig',False)
      figname = kwargs.get('figname','model.png')
      assert len(self.vp_array) != 1

      if type=='abs':
        vp_field  = self.vp_array
        vs_field  = self.vs_array
        rho_field = self.rho_array
      elif type=='abs_diff':
        vp_field  = self.dvp_abs
        vs_field  = self.dvs_abs
        rho_field = self.drho_abs
      elif type=='rel_diff':
        vp_field  = self.dvp_rel
        vs_field  = self.dvs_rel
        rho_field = self.drho_rel

      plt.style.use('mystyle')
      fig,axes = plt.subplots(1,3,figsize=(15,12))
      axes[0].set_title('Vp')
      axes[0].set_xlabel('theta (degrees)')
      axes[0].set_ylabel('radius(km)')
      a0 = axes[0].imshow(vp_field,aspect='auto',cmap='hot')
      c0 = plt.colorbar(a0,ax=axes[0],orientation='horizontal')
      axes[1].set_title('Vs')
      axes[1].set_xlabel('theta (degrees)')
      axes[1].set_ylabel('radius(km)')
      a1 = axes[1].imshow(vs_field,aspect='auto',cmap='hot')
      c1 = plt.colorbar(a1,ax=axes[1],orientation='horizontal')
      axes[2].set_title('Density')
      axes[2].set_xlabel('theta (degrees)')
      axes[2].set_ylabel('radius(km)')
      a2 = axes[2].imshow(rho_field,aspect='auto',cmap='hot')
      c2 = plt.colorbar(a2,ax=axes[2],orientation='horizontal')

      if type == 'abs' or type == 'abs_diff':
         c0.set_label('dVp (km/s)')
         c1.set_label('dVs (km/s)')
         c2.set_label('drho (kg/m^3)')
      else:
         c0.set_label('dVp (%)')
         c1.set_label('dVs (%)')
         c2.set_label('drho (%)')

      if contour == True:
         v = (-15.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0)
         axes[0].contour(vp_field,v,colors='k',alpha=0.5)
         axes[1].contour(vs_field,v,colors='k',alpha=0.5)
         axes[2].contour(rho_field,v,colors='k',alpha=0.5)

      if savefig:
         plt.savefig(figname)
      else:
         plt.show()

   def scale_T(self, sf=1.0):
      self.T *= sf
      self.T_adiabat *= sf

   def add_T(self,dT=0.0):
      self.T += dT
      self.T_adiabat += dT

   def gaussian_filter(self,sigma_x=0.0,sigma_y=0.0):
      '''
      Applies a gaussian filter to the seismic velocity field to mimic
      the loss of spatial resolution introduced in tomographic imaging
      '''
      from scipy.ndimage.filters import gaussian_filter

      #filter absolute perturbations
      dvp_filtered  = gaussian_filter(self.dvp_abs,sigma=[sigma_x,sigma_y])
      dvs_filtered  = gaussian_filter(self.dvs_abs,sigma=[sigma_x,sigma_y])
      drho_filtered = gaussian_filter(self.drho_abs,sigma=[sigma_x,sigma_y])
      self.dvp_abs  = dvp_filtered
      self.dvs_abs  = dvs_filtered
      self.drho_abs = drho_filtered

      #filter relative perturbations
      dvp_filtered  = gaussian_filter(self.dvp_rel,sigma=[sigma_x,sigma_y])
      dvs_filtered  = gaussian_filter(self.dvs_rel,sigma=[sigma_x,sigma_y])
      drho_filtered = gaussian_filter(self.drho_rel,sigma=[sigma_x,sigma_y])
      self.dvp_rel  = dvp_filtered
      self.dvs_rel  = dvs_filtered
      self.drho_rel = drho_filtered

   def plot_T(self):
      plt.imshow(self.T,cmap='hot')
      plt.colorbar()
      plt.show()

   def plot_1d_profiles(self,plot_all=True,var='all'):
      if plot_all == True:
         fig,ax = plt.subplots(figsize=(6,8))
         plt.plot(self.vp1d, self.rad_km[::-1],'r',label='Vp')
         plt.plot(self.vs1d, self.rad_km[::-1],'b',label='Vs')
         plt.plot(self.rho1d/1000.0, self.rad_km[::-1],'k',label='density')
         plt.xlim([2.0,16.0])
         plt.xlabel('velocity (km/s), density (kg/m^3)')
      else:
         if var == 'vp':
            fig,ax = plt.subplots(figsize=(6,8))
            plt.plot(self.vp1d, self.rad_km[::-1],'k',label='Vp')
            plt.xlabel('velocity (km/s)')
         if var == 'vs':
            fig,ax = plt.subplots(figsize=(6,8))
            plt.plot(self.vs1d, self.rad_km[::-1],'k',label='Vs')
            plt.xlabel('velocity (km/s)')
         if var == 'density':
            fig,ax = plt.subplots(figsize=(6,8))
            plt.plot(self.rho1d/1000.0, self.rad_km[::-1],'k',label='density')
            plt.xlabel('density (kg/m^3)')

      plt.ylim([3500.0,6371.0])
      plt.ylabel('radius (km)')
      plt.legend()
      plt.grid()
      plt.show()

   def plot_tables(self):
      fig,axes = plt.subplots(1,3,figsize=(15,6))
      axes[0].imshow(self.vp_table.T,aspect='auto')
      axes[1].imshow(self.vs_table.T,aspect='auto')
      axes[2].imshow(self.rho_table.T,aspect='auto')
      axes[0].contour(self.vp_table.T,colors='k')
      axes[1].contour(self.vs_table.T,colors='k')
      axes[2].contour(self.rho_table.T,colors='k')
      plt.show()

   def write_points(self,output_name,one_file=False):
      '''
      writes a points file that can be used in the 2D to 3D conversion software
      #NOTE 4/21/16: This is outdated and will be replaced by a method of the
                     velocity_model class, which outputs SES3D readable files.

      Params:
      output_name : name of the file you want to write
      '''

      def polar_2_cart(radius,theta,theta_deg=True):
         '''
         takes polar coordinates and returns cartesian.theta should be in degrees
         '''
         if theta_deg == True:
           theta = np.radians(theta)

         x = radius*np.cos(theta) 
         y = radius*np.sin(theta)
         return(x,y) 

      if one_file:
         f = open(output_name,'w')
         for i in range(0,len(self.rad_km)):
            for j in range(0,len(self.theta)):
               x, y = polar_2_cart(self.rad_km[i],self.theta[j])
               o1   = x
               o2   = y
               o3   = self.vp_array[(self.npts_rad-1)-i,j]
               o4   = self.vs_array[(self.npts_rad-1)-i,j]
               o5   = self.rho_array[(self.npts_rad-1)-i,j]/1000.0
               o6   = self.dvp_abs[(self.npts_rad-1)-i,j]
               o7   = self.dvs_abs[(self.npts_rad-1)-i,j]
               o8   = self.drho_abs[(self.npts_rad-1)-i,j]/1000.0
               line = '{} {} {} {} {} {} {} {}'.format(o1,o2,o3,o4,o5,o6,o7,o8)
               f.write(line+'\n')
      else:
         f_dvp = open(output_name+'_dvp','w')
         f_dvs = open(output_name+'_dvs','w')
         for i in range(0,len(self.rad_km)):
            for j in range(0,len(self.theta)):
               x, y = polar_2_cart(self.rad_km[i],self.theta[j])
               dvp_here = self.dvp_rel[(self.npts_rad-1)-i,j]
               dvs_here = self.dvs_rel[(self.npts_rad-1)-i,j]
               f_dvp.write('{} {} {}'.format(y,x,dvp_here)+'\n')
               f_dvs.write('{} {} {}'.format(y,x,dvs_here)+'\n')

   def write_axisem_1d(self,name):
      import subprocess

      rad = self.rad_km[::-1]*1000.0
      rho = self.rho1d
      vpv = self.vp1d*1000.0
      vsv = self.vs1d*1000.0

      core_file = '/geo/home/romaguir/utils/core.bm'

      f = open(name,'w')
      f.write('ANELASTIC       F'+'\n')
      f.write('ANISOTROPIC     F'+'\n')
      f.write('UNITS        m'+'\n')
      f.write('COLUMNS       radius      rho      vpv      vsv'+'\n')

      #f_core = open(core_file,'r')

      #write mantle
      for i in range(0,len(rad)):
         line = '{:7f} {:8.4f} {:8.4f} {:8.4f}'.format(rad[i],rho[i],vpv[i],vsv[i])
         f.write(line+'\n')

      #write core
      with open(core_file,'r') as infile:
         for line in infile:
            f.write(line)

      #cmd = "cat "+core_file+" >> "+name
      #subprocess.call(cmd,shell=True)

   def rotate_about_axis(self,**kwargs):
      '''
      This function extrapolated the 2D model data into a 3D section of the earth,
      by rotation about an axis of symmetry.  The default
      
      params:
      radmin: minimum radius of 3D model (km)          default: 3493.0
      radmax: maximum radius of 3D model (km)          default: 6371.0
      latmin: minimum latitude of 3D model (deg)     default: -10.0
      latmax: maximum latitude of 3D model (deg)     default: 10.0
      lonmin: minimum longitude of 3D model (deg)    default: -10.0
      lonmax: maximum longitude of 3D model (deg)    default: 10.0
      drad: step size in radius of 3D model (km)     default: 20.0
      dlat: step size in latitude of 3D model (deg)  default: 1.0
      dlon: step size in longitude of 3D model (deg) default: 1.0
      axis_lat: latitude of rotation axis (deg)      default: 0.0
      axis_lon: longitude of rotation axis (deg)     default: 0.0

      max_dist_from_axis: the maximum distances away from the rotation
                          axis for which a value will be interpolated
                          default: 10.0
      type: return models with absolute or relative perturbations.
                          one of 'abs' or 'rel'

      returns: dvp3d, dvs3d, drho3d (three instances of the model_3d class)
      '''
      from seis_tools.models.velocity_conversion import bilin_interp
      
      radmin = kwargs.get('rmin',3493.0)
      radmax = kwargs.get('rmax',6371.0)
      latmin = kwargs.get('latmin',-10.0)
      latmax = kwargs.get('latmax',10.0)
      lonmin = kwargs.get('lonmin',-10.0)
      lonmax = kwargs.get('lonmax',10.0)
      drad = kwargs.get('drad',20.0)
      dlat = kwargs.get('dlat',1.0)
      dlon = kwargs.get('dlon',1.0)
      axis_lat = kwargs.get('axis_lat',0.0)
      axis_lon = kwargs.get('axis_lon',0.0)
      max_dist_from_axis = kwargs.get('max_dist_from_axis',10)
      type = kwargs.get('type','abs')

      print radmin, radmax
      #initialize 3d fields as empty models_3d objects---------------------------
      dvp3d = model_3d(radmin=radmin,radmax=radmax,latmin=latmin,latmax=latmax,
                       lonmin=lonmin,lonmax=lonmax,drad=drad,dlat=dlat,dlon=dlon)
      dvs3d = model_3d(radmin=radmin,radmax=radmax,latmin=latmin,latmax=latmax,
                       lonmin=lonmin,lonmax=lonmax,drad=drad,dlat=dlat,dlon=dlon)
      drho3d = model_3d(radmin=radmin,radmax=radmax,latmin=latmin,latmax=latmax,
                       lonmin=lonmin,lonmax=lonmax,drad=drad,dlat=dlat,dlon=dlon)

      _rad = dvp3d.rad
      _lat = np.radians(dvp3d.lat)
      _lon = np.radians(dvp3d.lon)

      #absolute (km/s, kg/m3) or relative (%) perturbations----------------------
      if type == 'rel':
         _dvp = self.dvp_rel
         _dvs = self.dvs_rel
         _drho = self.drho_rel
      elif type == 'abs':
         _dvp = self.dvp_abs
         _dvs = self.dvs_abs
         _drho = self.drho_abs

      #flip 2d matrices
      dvp_mat = _dvp.T
      dvp_mat = np.fliplr(dvp_mat)
      dvs_mat = _dvs.T
      dvs_mat = np.fliplr(dvs_mat)
      drho_mat = _drho.T
      drho_mat = np.fliplr(drho_mat)

      #THIS IS FOR CELL DATA
      #fill out each field-------------------------------------------------------
      for i in range(0,len(_lon)-1):
         for j in range(0,len(_lat)-1):
            for k in range(0,len(_rad)-1):
            
               arc_distance = np.arccos(np.sin(_lat[i]) * np.sin(axis_lat)+
                                        np.cos(_lat[i]) * np.cos(axis_lat)*
                                        np.cos(_lon[j] - axis_lon))

               if np.degrees(arc_distance) <= max_dist_from_axis:

                  ir_2d  = int(_rad[k]-radmin)/self.dpr_km
                  it_2d =  int(arc_distance/self.dpth)  
                  dth_unit  = (arc_distance - it_2d *self.dpth)/self.dpth
                  dr_unit   = (_rad[k] - radmin - ir_2d * self.dpr_km)/self.dpr_km

                  #map 3d grid points
                  dvp3d.data[k,j,i] = bilin_interp(dvp_mat,it_2d,ir_2d,dth_unit,
                                                   dr_unit,self.npts_theta,self.npts_rad)
                  dvs3d.data[k,j,i] = bilin_interp(dvs_mat,it_2d,ir_2d,dth_unit,
                                                   dr_unit,self.npts_theta,self.npts_rad)
                  drho3d.data[k,j,i] = bilin_interp(drho_mat,it_2d,ir_2d,dth_unit,
                                                   dr_unit,self.npts_theta,self.npts_rad)
      #THIS IS FOR POINT DATA
      #fill out each field-------------------------------------------------------
      #'''
      for i in range(0,len(_lon)):
         for j in range(0,len(_lat)):
            for k in range(0,len(_rad)):
            
               arc_distance = np.arccos(np.sin(_lat[i]) * np.sin(axis_lat)+
                                        np.cos(_lat[i]) * np.cos(axis_lat)*
                                        np.cos(_lon[j] - axis_lon))

               if np.degrees(arc_distance) <= max_dist_from_axis:

                  ir_2d  = int(_rad[k]-radmin)/self.dpr_km
                  it_2d =  int(arc_distance/self.dpth)  
                  dth_unit  = (arc_distance - it_2d *self.dpth)/self.dpth
                  dr_unit   = (_rad[k] - radmin - ir_2d * self.dpr_km)/self.dpr_km

                  #map 3d grid points
                  dvp3d.data_pts[k,j,i] = bilin_interp(dvp_mat,it_2d,ir_2d,dth_unit,
                                                      dr_unit,self.npts_theta,self.npts_rad)
                  dvs3d.data_pts[k,j,i] = bilin_interp(dvs_mat,it_2d,ir_2d,dth_unit,
                                                      dr_unit,self.npts_theta,self.npts_rad)
                  drho3d.data_pts[k,j,i] = bilin_interp(drho_mat,it_2d,ir_2d,dth_unit,
                                                      dr_unit,self.npts_theta,self.npts_rad)
      #'''
      return dvp3d,dvs3d,drho3d

   def prem_pressure_to_depth(self):
      prem = models_1d.prem()
      interp_r = interp1d(prem.p,prem.r)
      self.rad_km = interp_r(self.P1d*1e5)
      self.rad_km[0] = 6371.0
                  

def polar_2_cart(radius,theta,theta_deg=True):
         '''
         takes polar coordinates and returns cartesian.theta should be in degrees
         '''
         if theta_deg == True:
           theta = np.radians(theta)

         x = radius*np.cos(theta)
         y = radius*np.sin(theta)
         return(x,y)

def write_ses3d_blocks(model_3d,save_dir,**kwargs):
   '''
   takes an instance of the model_3d class and writes ses3d block files
   for it's domain

   params:
   model_3d: instance of model_3d class
   save_dir: save directory

   **kwargs
   n_subdomains: number of subdomains for ses3d blocks (default 1)
   '''
   n_subdomains = kwargs.get('n_subdomains',1)
   b1 = open('block_x','w') #colatitude (degrees)
   b2 = open('block_y','w') #longitude  (degrees)
   b3 = open('block_z','w') #radius     (km)
    
   colat = model_3d.colat[::-1]
   lon = model_3d.lon
   rad = model_3d.rad

   #write file headers
   b1.write(str(n_subdomains)+'\n')
   b2.write(str(n_subdomains)+'\n')
   b3.write(str(n_subdomains)+'\n')
   b1.write(str(len(colat))+'\n')
   b2.write(str(len(lon))+'\n')
   b3.write(str(len(rad))+'\n')

   for i in range(0,len(colat)):
      b1.write(str(colat[i])+'\n')
   for i in range(0,len(lon)):
      b2.write(str(lon[i])+'\n')
   for i in range(0,len(rad)):
      b3.write(str(rad[i])+'\n')

def write_ses3d_perturbations(model_3d_list,save_dir,**kwargs):
   '''
   takes a list of [dvp,dvs,drho] (each an instance of the model_3d class
   ** make sure that the models are given in the proper order

   params:
   model_3d: list of models [dvp,dvs,drho]
   save_dir: save directory

   **kwargs
   n_subdomains: number of subdomains for ses3d blocks (default 1)
   '''

   #check that units are correct
   #for model in model_3d_list: 
   #   if max(model.data) >= 10.0:
         #raise ValueError('Check units. May need to divide by 1000')

   #total number of points
   dvp_model = model_3d_list[0]
   dvs_model = model_3d_list[1]
   drho_model = model_3d_list[2]


   l = (len(dvp_model.lat)-1)*(len(dvp_model.lon)-1)*(len(dvp_model.rad)-1)

   #initialize files
   n_subdomains = kwargs.get('n_subdomains',1)
   f1 = open(save_dir+'dvp','w')
   f2 = open(save_dir+'dvsv','w')
   f3 = open(save_dir+'dvsh','w')
   f4 = open(save_dir+'drho','w')
   f1.write(str(n_subdomains)+'\n')
   f1.write(str(l)+'\n')
   f2.write(str(n_subdomains)+'\n')
   f2.write(str(l)+'\n')
   f3.write(str(n_subdomains)+'\n')
   f3.write(str(l)+'\n')
   f4.write(str(n_subdomains)+'\n')
   f4.write(str(l)+'\n')

   #write values
   for i in range(0,len(dvp_model.lat)-1):
      for j in range(0,len(dvp_model.lon)-1):
         for k in range(0,len(dvp_model.rad)-1):
            line = '{}'.format(dvp_model.data[k,i,j])
            f1.write(line+'\n')
            line = '{}'.format(dvs_model.data[k,i,j])
            f2.write(line+'\n')
            line = '{}'.format(dvs_model.data[k,i,j])
            f3.write(line+'\n')
            line = '{}'.format(drho_model.data[k,i,j])
            f4.write(line+'\n')


def bilin_interp(grid2d, ith, ir, dth, dr, nth, nr):
   '''
   This function finds an interpolated value in the 2d plume grid, and returns
   the value. It's much faster than scipy.interpolate.interp2d

   Takes as arguments the 2d grid of plume perturbations, the index in theta,
   the index in radius, step size in theta and radius, and number of points
   in theta and radius.
   '''
   value = grid2d[ith,ir]*(1-dth)*(1-dr) + grid2d[ith+1,ir]*dth*(1-dr)
   value += grid2d[ith,ir+1]*(1-dth)*dr + grid2d[ith+1,ir+1]*dth*dr
   return value

def write_h5py_lookuptable(path_to_file,composition,**kwargs):
   '''
   This function reads a .tab lookup table and writes it to a new or
   existing h5py file

   Params
   path_to_file: string that gives the path to the .tab file
   composition: name of composition model

   **kwargs
   new: True/False : whether to write a new file or append to an existing file
   fname: if new = True, give the file a name
   location: if new = False, give the file you with to append to
   '''
   new = kwargs.get('new',False)
   fname = kwargs.get('fname','none')
   location = kwargs.get('location','none')

   #parse the file and pull out important information
   info = open(path_to_file,'r')
   junk = info.readline().strip()
   junk = info.readline().strip()
   junk = info.readline().strip()
   junk = info.readline().strip()
   P0 = info.readline().strip()
   dP = info.readline().strip()
   nP = info.readline().strip()
   junk = info.readline().strip()
   T0 = info.readline().strip()
   dT = info.readline().strip()
   nT = info.readline().strip()
   info.close()
   nP=int(nP)
   nT=int(nT)
   dP=float(dP)
   dT=float(dT)
   P0=float(P0)
   T0=float(T0)
   v_file  = np.loadtxt(path_to_file,skiprows=13)
   rho_vec = v_file[:,4]
   vp_vec  = v_file[:,8]
   vs_vec  = v_file[:,9]

   vp_table  = np.zeros((nT,nP))
   vs_table  = np.zeros((nT,nP))
   rho_table = np.zeros((nT,nP))

   k = 0
   for i in range(0,nT):
       for j in range(0,nP):
           vp_table[i,j]  = vp_vec[k]
           vs_table[i,j]  = vs_vec[k]
           rho_table[i,j] = rho_vec[k]
           k+=1

   if new == True: 
      f = h5py.File(fname,'w')
   elif new == False:
      f = h5py.File(location,'r+')

   grp = f.create_group(composition)
   grp.create_dataset('vp',data=vp_table)
   grp.create_dataset('vs',data=vs_table)
   grp.create_dataset('rho',data=rho_table)
   grp.create_dataset('dP',data=dP)
   grp.create_dataset('dT',data=dT)
   grp.create_dataset('nP',data=nP)
   grp.create_dataset('nT',data=nT)
   grp.create_dataset('P0',data=P0)
   grp.create_dataset('T0',data=T0)

   f.close()

def plot_models_list(models_list,var_to_plot,return_axis=False):
   '''
   plots 1d profiles for a list of velocity_model objects

   args:
   models_list: list of velocity_model objects
   var_to_plot: 'vp', 'vs', or 'rho'
   '''
   from matplotlib.colors import Normalize
   plt.style.use('mystyle')
   fig,ax = plt.subplots()
   cmap = plt.get_cmap('hot')
   vals = np.linspace(0,1,len(models_list)*1.5)
   norm = Normalize()
   colors = cmap(norm(vals))

   i = 0
   if var_to_plot == 'vp':
      for m in models_list:
         ax.plot(m.vp1d,m.rad_km[::-1],label='{:4.0f} {}'.format(m.pot_T,'K'),color=colors[i])
         i += 1
   elif var_to_plot == 'vs':
      for m in models_list:
         ax.plot(m.vs1d,m.rad_km[::-1],label='{:4.0f} {}'.format(m.pot_T,'K'),color=colors[i])
         i += 1
   elif var_to_plot == 'rho':
      for m in models_list:
         ax.plot(m.rho1d,m.rad_km[::-1],label='{:4.0f} {}'.format(m.pot_T,'K'),color=colors[i])
         i += 1
   
   ax.set_ylabel('radius (km)')
   ax.set_xlabel('velocity (km/s)')
   ax.legend(loc='lower left')
   ax.grid()
   
   if return_axis == False:
      plt.show()
   else:
      return ax

def bm_to_tvel(name):
   f = np.loadtxt(name+'.bm',skiprows=4)
   rad_m = f[:,0]
   rho_kgm3 = f[:,1]
   vp_ms2 = f[:,2]
   vs_ms2 = f[:,3]

   depth_km = 6371.0-(rad_m/1000.0)
   rho_gcm3 = rho_kgm3/1000.0
   vp_kms2 = vp_ms2/1000.0
   vs_kms2 = vs_ms2/1000.0

   out_file = open(name+'.tvel','w')
   out_file.write('{} {}'.format(name,'- P'))
   out_file.write('\n')
   out_file.write('{} {}'.format(name,'- S'))
   out_file.write('\n')

   for i in range(0,len(depth_km)):
      out_file.write('{:6.4f} {:6.4f} {:6.4f} {:6.4f}'.format(depth_km[i],vp_kms2[i],vs_kms2[i],rho_gcm3[i]))
      out_file.write('\n')

def write_plume_gmt_files(velocity_model,plume_name,remove_noise=True,theta_max='none',depth_max='none'):
   '''
   writes files for plotting delta T, delta Vs plots in GMT

   args:
      velocity_model: an instance of the velocity_model class
      plume_name: what to call the output files
      remove_noise: removes positive values of dVs and negative values of delta T
      theta_max: if not = 'none', then it will truncate plume at a given value theta
      depth_max: if not = 'none', then truncate plume below given depth (km)
   '''
   fout1 = open(plume_name+'_dT.dat','w')
   fout2 = open(plume_name+'_dVs.dat','w')
 
   if remove_noise:
      for i in range(0,velocity_model.npts_rad):
         for j in range(0,velocity_model.npts_theta):
            if velocity_model.dvs_rel[i,j] > 0:
               velocity_model.dvs_rel[i,j] = 0
            if velocity_model.delta_T[i,j] < 0:
               velocity_model.delta_T[i,j] = 0

   if theta_max != 'none':
      for i in range(0,velocity_model.npts_rad):
         for j in range(0,velocity_model.npts_theta):
            if velocity_model.theta[j] > theta_max:
               velocity_model.dvs_rel[i,j] = 0.0
               velocity_model.delta_T[i,j] = 0.0

   if depth_max != 'none':
      rad_trunc = 6371.0-depth_max
      for i in range(0,velocity_model.npts_rad):
         for j in range(0,velocity_model.npts_theta):
             if velocity_model.rad_km[::-1][i] < rad_trunc:
               velocity_model.dvs_rel[i,j] = 0.0
               velocity_model.delta_T[i,j] = 0.0

   for i in range(0,velocity_model.npts_rad):
       for j in range(0,velocity_model.npts_theta):
          fout1.write('{} {} {}'.format(velocity_model.theta[j]+90.0,
                                        velocity_model.rad_km[::-1][i],
                                        velocity_model.delta_T[i,j])+'\n')
          fout2.write('{} {} {}'.format(-1.0*velocity_model.theta[j]+90.0,
                                        velocity_model.rad_km[::-1][i],
                                        velocity_model.dvs_rel[i,j])+'\n')

   fout1.close()
   fout2.close()
