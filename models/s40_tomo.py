import numpy as np

def write_s40_layer(depth_range,file_name,percent_dv,lat_range=(-20,20),lon_range=(-20,20)):

   '''
   write a single layer input file for the s40 filter

   args--------------------------------------------------------------------------
   depth_range: tuple of the depth range of the layer
   file_name: obvious
   percent_dv: obvious
   lat_range: latitude range of heterogeneity (default=(-20,20))
   lon_range: longiutde range of heterogeneity (default=(-20,20))
   '''

   #definitions
   lats = np.arange(-90,91)
   lons = np.arange(0,360)
   lat_min = lat_range[0]
   lat_max = lat_range[1]
   lon_min = lon_range[0]
   lon_max = lon_range[1]

   #write header
   output = open(file_name,'w')
   output.write(str(depth_range[0])+'\n')
   output.write(str(depth_range[1])+'\n')

   #write layer
   for lat in lats:
      for lon in lons:
         if lat >= lat_min and lat <= lat_max and lon >= lon_min and lon <= lon_max:
            line = '{} {} {}'.format(lat,lon,percent_dv)
            output.write(line+'\n')
         else:
            line = '{} {} {}'.format(lat,lon,0.0)
            output.write(line+'\n')

def get_rspl_coeff():

    #The following are the 21 vertical spline coefficients used in S40RTS

    coeffs = [-1.00000,-0.78631,-0.59207,-0.41550,-0.25499,
              -0.10909, 0.02353, 0.14409, 0.25367, 0.35329,
               0.44384, 0.52615, 0.60097, 0.66899, 0.73081,
               0.78701, 0.83810, 0.88454, 0.92675, 0.96512,
               1.00000]

