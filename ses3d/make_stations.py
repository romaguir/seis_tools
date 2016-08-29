import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#=======================================================
# Create grid of receivers for SES3d
#=======================================================

depth     = 0.0
rec_names = []
rec_coors = []

#main station grid
d_lon    = 0.5
d_lat    = 0.5
lon_0    = -15.0
lon_e    = 15.0
colat_0  = 75
colat_e  = 105
lon   = np.arange(lon_0,lon_e+(1*d_lon),d_lon) 
colat = np.arange(colat_0,colat_e+(1*d_lat),d_lat)

#sparse station grid
d_lon2    = 3.0 
d_lat2    = 3.0
lon2_0    = -30.0
lon2_e    = 30.0
colat2_0    = 60.0
colat2_e    = 120.0
lon2  = np.arange(lon2_0,lon2_e+(1*d_lon2),d_lon2)
colat2  = np.arange(colat2_0,colat2_e+(1*d_lat2),d_lat2)

total_receivers = len(lon)*len(colat) + len(lon2)*len(colat2)

m = Basemap(projection='ortho',lon_0=0.0,lat_0=0.0)
m.drawcoastlines()
parallels = np.arange(-90,90,15)
m.drawparallels(parallels)
meridians = np.arange(-90,90,15)
m.drawmeridians(meridians)

count = 1
for l in lon:
   for c in colat:
      rec_name =  '{}''{:_>9}'.format('REC',str(count))
      rec_coor = '{} {} {}'.format(c, l, depth)
      rec_names.append(rec_name) 
      rec_coors.append(rec_coor)
      count += 1
      x1,y1 = m(l,90.0-c)
      m.scatter(x1,y1)

for l in lon2:
   for c in colat2:
      if c < 75.0 or c > 105.0 or l < -15.0 or l > 15.0:
         rec_name =  '{}''{:_>9}'.format('REC',str(count))
         rec_coor = '{} {} {}'.format(c, l, depth)
         rec_names.append(rec_name) 
         rec_coors.append(rec_coor)
         count += 1
         x1,y1 = m(l,90.0-c)
         m.scatter(x1,y1)

#write the station file
f = open('recfile_1','w')
f.write(str(len(rec_names))+'\n')

for i in range(0,len(rec_names)): 
   f.write(rec_names[i]+'\n')
   f.write(rec_coors[i]+'\n')

plt.show()
