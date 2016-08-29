import numpy as np
from numpy import cos, sin, pi
from tvtk.api import tvtk
from mayavi.scripts import mayavi2

def generate(phi=None, theta=None, rad=None):
    # Default values for the spherical section
    #if rad is None: rad = np.linspace(3490.0,6371.0,20)
    if rad is None: rad = np.linspace(1.0,2.0,50)
    if phi is None: phi = np.linspace(0,2*pi,50)
    if theta is None: theta = np.linspace(0.0,pi,50)

    # Find the x values and y values for each plane.
    #x_plane = (cos(phi)*theta[:,None]).ravel()
    #y_plane = (sin(phi)*theta[:,None]).ravel()
    #print "len x_plane = ", len(x_plane)

    # Allocate an array for all the points.  We'll have len(x_plane)
    # points on each plane, and we have a plane for each z value, so
    # we need len(x_plane)*len(z) points.
    len_points =  len(theta)*len(phi)*len(rad)
    points = np.empty([len_points,3])

    # Loop through the points and fill them with the
    # correct x,y,z values.
    count = 0 

    print 'rad',rad
    print 'theta',theta
    print 'phi',phi

    for i in range(0,len(theta)-1):
       for j in range(0,len(phi)-1):
          for k in range(0,len(rad)-1):
             x = rad[k]*cos(phi[j])*sin(theta[i])
             y = rad[k]*sin(phi[j])*sin(theta[i])
             z = rad[k]*cos(theta[i])
             points[count,:] = x,y,z
             count += 1

    return points
