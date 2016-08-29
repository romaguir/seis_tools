import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

def cart2polar(x,y):
   '''
   Given x,y in cartesian coordinates, returns r, theta
   '''

   r = np.sqrt(x**2+y**2)
   theta = np.arctan2(y,x)

   return r,theta

def interp_plume_dv(plume_model,pts_file,fname_out='plume.xy',zero_center=True):
   '''
   Takes a plume model file, interpolates it to a new grid, and writes a new file
 
   args--------------------------------------------------------------------------
   plume_model: the plume model file path.  cartesian coordinate system
   pts_file: a file containing the points to interpolate to
   fname_out: the name of the output file
   '''

   f1 = np.loadtxt(plume_model) 
   x = f1[:,0]
   y = 6371.0 - f1[:,1] 
   v = f1[:,2]

   f2 = np.loadtxt(pts_file,skiprows=1)
   x_new = f2[:,0]
   y_new = f2[:,1]


   x_mirror = x*-1.0 
   x_pts = np.hstack((x,x_mirror))
   y_pts = np.hstack((y,y))
   v_pts = np.hstack((v,v))

   if not zero_center: 
      x_pts += max(x_new)/2
   
   interpolator = NearestNDInterpolator((x_pts,y_pts),v_pts)

   v_new = interpolator(x_new,y_new)

   f3 = open(fname_out,'w')
   for i in range(0,len(x_new)):
      f3.write('{} {} {}'.format(x_new[i],y_new[i],v_new[i])+'\n') 

   f3.close()
