import numpy as np
from seis_tools.ses3d import rotation

def rotate_about_axis(tr,lon_0=60.0,lat_0=0.0,degrees=0):
   '''
   Rotates the source and receiver of a trace object around an
   arbitrary axis.
   '''

   alpha = np.radians(degrees)
   lon_s = tr.stats.sac['evlo']
   lon_r = tr.stats.sac['stlo']
   colat_s = 90.0-tr.stats.sac['evla']
   colat_r = 90.0-tr.stats.sac['stla']
   colat_0 = 90.0-lat_0

   x_s = lon_s - lon_0
   y_s = colat_0 - colat_s
   x_r = lon_r - lon_0
   y_r = colat_0 - colat_r


   #rotate receiver
   tr.stats.sac['stla'] = 90.0-colat_0+x_r*np.sin(alpha) + y_r*np.cos(alpha)
   tr.stats.sac['stlo'] = lon_0+x_r*np.cos(alpha) - y_r*np.sin(alpha)

   #rotate source
   tr.stats.sac['evla'] = 90.0-colat_0+x_s*np.sin(alpha) + y_s*np.cos(alpha)
   tr.stats.sac['evlo'] = lon_0+x_s*np.cos(alpha) - y_s*np.sin(alpha)

def cartesian_to_spherical(x,degrees=True,normalize=False):
   '''
   Coverts a cartesian vector in R3 to spherical coordinates
   '''
   r = np.linalg.norm(x)
   theta = np.arccos(x[2]/r)
   phi = np.arctan2(x[1],x[0])

   if degrees:
      theta = np.degrees(theta)
      phi = np.degrees(phi)

   s = [r,theta,phi]

   if normalize:
      s /= np.linalg.norm(s)

   return s

def spherical_to_cartesian(s,degrees=True,normalize=False):
   '''
   Takes a vector in spherical coordinates and converts it to cartesian.
   Assumes the input vector is given as [radius,colat,lon] 
   '''

   if degrees:
      s[1] = np.radians(s[1])  
      s[2] = np.radians(s[2])

   x1 = s[0]*np.sin(s[1])*np.cos(s[2])
   x2 = s[0]*np.sin(s[1])*np.sin(s[2])
   x3 = s[0]*np.cos(s[1])

   x = [x1,x2,x3]

   if normalize:
      x /= np.linalg.norm(x)
   return x

def find_rotation_vector(s1,s2):
   '''
   Takes two vectors in spherical coordinates, and returns the cross product,
   normalized to one.
   '''

   x1 = spherical_to_cartesian(s1,degrees=True,normalize=True) 
   x2 = spherical_to_cartesian(s2,degrees=True,normalize=True)
   n = np.cross(x1,x2)
   n /= np.linalg.norm(n)
   return n

def find_rotation_angle(s1,s2,degrees=True):
   '''
   Finds the angle between two vectors in spherical coordinates
   
   params:
   s1,s2: vectors in spherical coordinates

   returns
   '''
   x1 = spherical_to_cartesian(s1,degrees=True,normalize=True) 
   x2 = spherical_to_cartesian(s2,degrees=True,normalize=True)
   if degrees:
      return np.degrees(np.arccos(np.clip(np.dot(x1,x2),-1.0,1.0)))
   else:
      return np.arccos(np.clip(np.dot(x1,x2),-1.0,1.0))
   
