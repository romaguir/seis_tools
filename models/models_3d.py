import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from seis_tools.models import models_1d
from seis_tools.seispy.misc import find_rotation_vector

# 3D_model_class
class model_3d(object):
   '''
   a class for dealing with 3d global data

   args--------------------------------------------------------------------------
   latmin: minimum latitude of model (default = -10.0)
   latmax: maximum latitude of model (default = 10.0)
   lonmin: minimum longitude of model (default = -10.0)
   lonmax: maximum longitude of model (default = 10.0)
   radmin: minimum radius of model (default = 3494.0 km)
   radmax: maximum radius of model (default = 6371.0 km)
   dlat: latitude spacing (default = 1.0)
   dlon: longitude spacing (default = 1.0)
   drad: radius spacing (default = 20.0)

   kwargs------------------------------------------------------------------------
   region: select preset region
   '''

   def __init__(self,radmin=3494.0,radmax=6371.0,
                latmin=-10.0,latmax=10.0,
                lonmin= -10.0,lonmax=10.0,
                drad = 20.0,dlat = 1.0,dlon = 1.0,**kwargs):

      #check if using a preset region--------------------------------------------
      region = kwargs.get('region','None')
      if region == 'North_America_rfs':
         lonmin=232
         lonmax=294
         latmin=24
         latmax=52
         radmin=5000   
   
      #define model range--------------------------------------------------------
      nlat        = int((latmax-latmin)/dlat)+1
      nlon        = int((lonmax-lonmin)/dlon)+1
      nrad        = int((radmax-radmin)/drad)+1
      self.lon    = np.linspace(lonmin,lonmax,nlon)
      self.rad    = np.linspace(radmin,radmax,nrad)
      self.lat    = np.linspace(latmin,latmax,nlat)
      self.colat  = 90.0 - self.lat

      #knots---------------------------------------------------------------------
      #self.lat, self.lon and self.rad are defined as the grid points, and the
      #values are given inside the cells which they bound.  Hence, each axis has
      #one more grid point than cell value. Knots are the cell center points, so
      #there are the same number of knots as data points

      #3d field: data = cell data, data_pts = point data
      self.data = np.zeros((len(self.rad)-1,len(self.colat)-1,len(self.lon)-1))
      self.data_pts = np.zeros((len(self.rad),len(self.colat),len(self.lon)))
      self.dlat = dlat
      self.dlon = dlon
      self.drad = drad

   def rotate_3d(self,destination):
      '''
      solid body rotation of the mantle with respect to the surface.
      rotates the south pole to the specfied destination.

      args:
          destination: (lat,lon)
      '''
      s1 = (6371, 180, 0)
      s2 = (6371, 90-lat, lon)
      rotation_vec = find_rotation_vector(s1,s2)
      rotation_angle = find_rotation_angle(s2,s2)
      rotated_data = np.zeros(self.data.shape)
      for i in range(0,len(self.rad)):
          for j in range(0,len(self.colat)):
              for k in range(0,len(self.lon)):
                  old_coors = [self.rad[i],self.colat[j], self.lon[k]]
                  transformed_coors = rotate_coordinates(n=rotation_vec,
                                                         phi=rotation_angle,
                                                         colat=self.colat[j],
                                                         lon=self.lon[k])
                  transformed_colat = transformed_coors[0]
                  transformed_lon = transformed_coors[1]
                  rotated_data[i,j,k] = self.probe_data(rad,lat,lon)
                  rotated_data[i,j,k] = self.probe_data(self.rad[i],
                                                        90-self.lat[j],
                                                        self.lon[k])
      
      self.data = rotated_data

   def plot_3d(self,**kwargs):
      '''
      plots a 3d earth model using mayavi
      
      kwargs:
              grid_x: Draw the x plane (default False)
              grid_y: Draw the y plane (default False)
              grid_z: Draw the z plane (default False)
              earth:  Draw the earth outline and coastlines (default True)
              plot_quakes: Draw earthquakes in earthquake_list (default False)
              earthquake_list: a list of earthquake coordinates specified by 
                               (lat,lon,depth)
      '''
      import mayavi
      from mayavi import mlab
      from tvtk.api import tvtk
      from mayavi.scripts import mayavi2
      from spherical_section import generate
      from mayavi.sources.vtk_data_source import VTKDataSource
      from mayavi.sources.builtin_surface import BuiltinSurface
      from mayavi.modules.api import Outline, GridPlane, ScalarCutPlane

      draw_gridx = kwargs.get('grid_x',False)
      draw_gridy = kwargs.get('grid_y',False)
      draw_gridz = kwargs.get('grid_z',False)
      draw_earth = kwargs.get('earth',True)
      draw_quakes = kwargs.get('draw_quakes',False)
      earthquake_list = kwargs.get('earthquake_list','none')

      #build the spherical section
      dims = (len(self.rad)-1,len(self.lon)-1,len(self.colat)-1)
      pts = generate(phi=np.radians(self.lon),theta=np.radians(self.colat),rad=self.rad)
      sgrid = tvtk.StructuredGrid(dimensions=dims)
      sgrid.points = pts
      s = np.zeros(len(pts))

      #map data onto the grid
      count = 0
      for i in range(0,len(self.colat)-1):
         for j in range(0,len(self.lon)-1):
            for k in range(0,len(self.rad)-1):
            
               s[count] = self.data[k,i,j]
               sgrid.point_data.scalars = s
               sgrid.point_data.scalars.name = 'scalars'
               count += 1
  
      #use vtk dataset
      src = VTKDataSource(data=sgrid)

      #set figure defaults
      mlab.figure(bgcolor=(0,0,0))

      #outline
      mlab.pipeline.structured_grid_outline(src,opacity=0.3)

      #show grid planes
      if draw_gridx == True:
         gx = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gx.grid_plane.axis='x'
      if draw_gridy == True:
         gy = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gy.grid_plane.axis='y'
      if draw_gridz == True:
         gz = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gz.grid_plane.axis='z'

      #cutplane
      mlab.pipeline.scalar_cut_plane(src,plane_orientation='y_axes',
                                     colormap='jet',view_controls=False)

      #draw earth and coastlines
      if draw_earth == True:
         coastline_src = BuiltinSurface(source='earth',name='Continents')      
         coastline_src.data_source.radius = 6371.0
         coastline_src.data_source.on_ratio = 1
         mlab.pipeline.surface(coastline_src)

      #plot earthquakes
      if draw_quakes == True:
         print "Sorry, this doesn't work yet"
         lat_pts=earthquake_list[:,0]
         lon_pts=earthquake_list[:,1]
         rad_pts=6371.0-earthquake_list[:,2]
         theta_pts = np.radians(lat_pts)
         phi_pts   = np.radians(lon_pts)
       
         #convert point to cartesian
         x_pts = rad_pts*np.cos(phi_pts)*np.sin(theta_pts)
         y_pts = rad_pts*np.sin(phi_pts)*np.sin(theta_pts)
         z_pts = rad_pts*np.cos(theta_pts)
         eq_pts = mlab.points3d(x_pts,y_pts,z_pts,
                                scale_mode='none',
                                scale_factor=100.0,
                                color=(1,1,0))

      mlab.show()

   def find_index(self,lat,lon,depth):
      '''
      provided a latitude, longitude, and depth, finds the corresponding
      model index
      '''
      rad       = 6371.0-depth
      lat_min   = self.lat[0]
      lon_min   = self.lon[0]
      rad_min   = self.rad[0]

      lat_i = int((lat-lat_min)/self.dlat)
      lon_i = int((lon-lon_min)/self.dlon)
      rad_i = int((rad-rad_min)/self.drad)

      if lat_i >= len(self.lat): 
         print "latitude ", lat, " is outside the model"
      if lon_i >= len(self.lon): 
         print "longitude ", lon, " is outside the model"
      if rad_i >= len(self.rad): 
         print "depth", depth, " is outside the model"

      return rad_i, lat_i, lon_i

   def map_rf(self,pierce_dict,**kwargs):
      '''
      takes a seispy receiver function, cycles through the pierce
      dictionary and maps the values to the 3d model. alternatively
      takes just the pierce point dictionary instead of the receiver
      function object
  
      **kwargs-------------------------------------------------------------------
      rf: a seispy receiver function object, complete with pierce point dictionary
      pierce_dict: dictionary containing geographic pierce point information, and
                   receiver function amplitude
      '''
      #from seis_tools.models.models_3d import find_index
      #rf = kwargs.get('rf','None')
      #pierce_dict = kwargs.get('pierce_dict','None')

      #if rf != 'None':
      #   pts = rf.pierce
      #elif pierce_dict != 'None':
      #   print 'this worked apparently'
      #   pts = pierce_dict
      pts = pierce_dict     
      print 'whaaaa'

      for i in pts:
         lat   = i['lat']
         lon   = i['lon']
         depth = i['depth']
         amp   = i['amplitude']

         ind = self.find_index(lat,lon,depth)
         print lat,lon,depth,ind
 
         #map the value
         #self.data[ind] += amp
         self.data[ind] = 1

   def probe_data(self,rad,lat,lon,**kwargs):
      '''
       returns the value of the field at the point specified. 
   
       params:
       lat
       lon
       depth
      '''
      type = kwargs.get('type','point')
      if type == 'cell':
         p1    = self.rad[0:len(self.rad)-1]
         p2    = self.lat[0:len(self.lat)-1]
         p3    = self.lon[0:len(self.lon)-1]
      
         return interpn(points = (p1,p2,p3),
                        values = self.data,
                        xi = (rad,lat,lon),
                        bounds_error=False,
                        fill_value = 0.0)

      elif type == 'point':
         return interpn(points=(self.rad,self.lat,self.lon),
                        values=self.data_pts,
                        xi = (rad,lat,lon),
                        bounds_error=False,
                        fill_value = 0.0)

   def save(self,format,filename):
      '''
      save an instance of the 3d model class
      '''
      if format == 'pickle':
         pickle.dump(self,file(filename,'w'))

   def write_specfem_ppm(self,**kwargs):
       '''
       writes a 3d model to ppm format for specfem
       '''
       fname = kwargs.get('fname','model.txt')
       f = open(fname,'w')
       f.write('#lon(deg), lat(deg), depth(km), Vs-perturbation wrt PREM(%), Vs-PREM(km/s) \n')
       prem = models_1d.prem()

       #loop through model and write points (lat = inner, lon = middle, rad = outer)
       for i in range(0,len(self.rad)):
	   depth = 6371.0 - self.rad[::-1][i]
           prem_vs = 5.0 #prem.get_vs(depth)
           for j in range(0,len(self.lon)):
               lon = self.lon[j]
               for k in range(0,len(self.lat)):
                   lat = self.lat[k]
                   dv = self.data_pts[(len(self.rad)-(i+1)),j,k]
                   f.write('{} {} {} {} {}'.format(lon,lat,depth,dv,prem_vs)+'\n')

   def write_specfem_heterogen(self,**kwargs):
       '''
       writes 3d model to a 'heterogen.txt' to be used in specfem.
       '''
       fname = kwargs.get('fname','heterogen.txt')
       f = open(fname,'w')
       for i in range(0,len(self.rad)):
           for j in range(0,len(self.lat)):
               for k in range(0,len(self.lon)):
                   f.write('{}'.format(self.data_pts[i,j,k])+'\n')

def write_specfem_ppm(dvs_model3d,dvp_model3d,drho_model3d,**kwargs):
    '''
    writes a 3d model to ppm format for specfem
    '''
    fname = kwargs.get('fname','model.txt')
    f = open(fname,'w')
    f.write('#lon(deg), lat(deg), depth(km), dvs(%), dvp(%), drho(%) \n')

    #loop through model and write points (lat = inner, lon = middle, rad = outer)
    for i in range(0,len(dvs_model3d.rad)):
       depth = 6371.0 - dvs_model3d.rad[::-1][i]
       for j in range(0,len(dvs_model3d.lon)):
          lon = dvs_model3d.lon[j]
          for k in range(0,len(dvs_model3d.lat)):
             lat = dvs_model3d.lat[k]
             dvs = dvs_model3d.data_pts[(len(dvs_model3d.rad)-(i+1)),j,k]
             dvp = dvp_model3d.data_pts[(len(dvp_model3d.rad)-(i+1)),j,k]
             drho = drho_model3d.data_pts[(len(drho_model3d.rad)-(i+1)),j,k]
             f.write('{} {} {} {} {} {}'.format(lon,lat,depth,dvs,dvp,drho)+'\n')

def write_s40_filter_inputs(model_3d,**kwargs):
   '''
   takes in a 3d model and writes out the input files for the S40RTS tomographic filter.
   
   params:
   model_3d: instance of the model_3d class

   kwargs:
   n_layers: number of layers (i.e, spherical shells).
             one file will be written per layer
   lat_spacing : spacing in latitude
   lon_spacing : spacing in longitude
   model_name : model_name (string)
   save_dir : save directory (string)

   Each output file is of the form-----------------------------------------------

   layer depth_start
   layer depth_end
   lat, lon, val
   .
   .   
   .
   
   lat, lon, val
   ------------------------------------------------------------------------------
   '''
   n_layers    = kwargs.get('n_layers',64)
   lat_spacing = kwargs.get('lat_spacing',1.0)
   lon_spacing = kwargs.get('lon_spacing',1.0)
   model_name  = kwargs.get('model_name','none')
   save_dir    = kwargs.get('save_dir','./')
   type = kwargs.get('type','point')

   #initializations
   lat = np.arange(-90,90,lat_spacing)
   lon = np.arange(0,360.0,lon_spacing)
   depth =  np.linspace(0,2878,n_layers)
   lon_min = min(model_3d.lon)
   lon_max = max(model_3d.lon)
   lat_min = min(model_3d.lat)
   lat_max = max(model_3d.lat)
    

   for i in range(0,len(depth)-1):
      r1 = 6371.0 - depth[i]
      r2 = 6371.0 - depth[i+1]
      r_here = (r1+r2)/2.0

      #open file and write header
      out_name = str(model_name)+'.'+str(i)+'.dat' 
      output   = open(save_dir+'/'+out_name,'w')
      output.write(str(6371.0-r1)+'\n')
      output.write(str(6371.0-r2)+'\n')

      for j in range(0,len(lon)):
         for k in range(0,len(lat)):
             
            if (lon[j] >= lon_min and lon[j] <= lon_max and
                lat[k] >= lat_min and lat[k] <= lat_max):

               if type == 'point':
                  value = model_3d.probe_data(r_here,lat[k],lon[j],type='point')
                  value = value[0]
               elif type == 'cell':
                  value = model_3d.probe_data(r_here,lat[k],lon[j],type='cell')
                  value = value[0]

            else:
               value = 0.0

            line = '{} {} {}'.format(lat[k],lon[j],value)
            output.write(line+'\n')
