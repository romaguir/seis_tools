import numpy as np
from seis_tools.models import models_3d
from scipy.interpolate import RegularGridInterpolator

def read_dna13(plot_3d=False):
   DNAFILE='/home/romaguir/Desktop/DNA13/DNA13/DNA13.4phase'
   f = np.loadtxt(DNAFILE,skiprows=1)
   
   rad = f[:,0]
   theta = f[:,1]
   phi = f[:,2]
   dvp = f[:,3]
   dvsh = f[:,4]
   dvsv = f[:,5]
   
   #Determine topology (model is defined on a regular grid)
   radmin = np.min(rad)
   radmax = np.max(rad)
   drad = np.max(np.diff(rad))
   rad_axis = np.arange(radmin,radmax+drad,drad)
   thetamin = np.min(theta) #latitude
   thetamax = np.max(theta)
   dtheta = np.max(np.diff(theta))
   theta_axis = np.arange(thetamin,thetamax+dtheta,dtheta)
   phimin = np.min(phi)
   phimax = np.max(phi)
   dphi = np.max(np.diff(phi))
   phi_axis = np.arange(phimin,phimax+dphi,dphi)
   
   dvp_array = np.reshape(dvp,(len(rad_axis),len(theta_axis),len(phi_axis)),order='F')
   dvsh_array = np.reshape(dvsh,(len(rad_axis),len(theta_axis),len(phi_axis)),order='F')
   dvsv_array = np.reshape(dvsv,(len(rad_axis),len(theta_axis),len(phi_axis)),order='F')
   #note, order = 'F' means that the inner loop is the first index (i.e., radius loops fastest)
   
   #plot ?
   if plot_3d:
      dvp_3d = models_3d.model_3d(radmin = radmin,radmax=radmax+drad,drad=drad,
                                  latmin = thetamin, latmax = thetamax, dlat = dtheta,
                                  lonmin = phimin, lonmax = phimax, dlon = dphi)
   
      dvp_3d.data = dvp_array

   interp_dvp3d = RegularGridInterpolator(points = (rad_axis,theta_axis,phi_axis),
                                          values = dvp_array,
                                          fill_value = 0)
   interp_dvsh3d = RegularGridInterpolator(points = (rad_axis,theta_axis,phi_axis),
                                           values = dvsh_array,
                                           fill_value = 0)
   interp_dvsv3d = RegularGridInterpolator(points = (rad_axis,theta_axis,phi_axis),
                                           values = dvsv_array,
                                           fill_value = 0)
   
   return interp_dvp3d,interp_dvsh3d,interp_dvsv3d
