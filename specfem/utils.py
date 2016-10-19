import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def make_stations(network,latmin,latmax,lonmin,lonmax,dlat,dlon,plot=True,**kwargs):
    head = kwargs.get('head','RM')
    elev = kwargs.get('elev',0)
    dep  = kwargs.get('dep',0)

    lats = np.arange(latmin,latmax+dlat,dlat)
    lons = np.arange(lonmin,lonmax+dlon,dlon)
    lats_pts,lons_pts = np.meshgrid(lats,lons)

    if plot:
        m = Basemap(projection='hammer',lon_0=0,resolution='l')
        m.drawcoastlines()
        m.drawmeridians(np.arange(0,351,10))
        m.drawparallels(np.arange(-80,81,10))
        lons_pts2,lats_pts2 = m(lons_pts,lats_pts)
        m.scatter(lons_pts2,lats_pts2)
        plt.show()

    f = open('STATIONS','w')
    print 'total number of stations : ',len(lats_pts)
    st = 1
    for i in range(0,len(lats)):
        for j in range(0,len(lons)):
            f.write('{}{:04d} {} {:5.2f} {:5.2f} {:5.2f} {:5.2f}'.format(head,st,network,lats[i],lons[j],elev,dep)+'\n')
            st += 1
