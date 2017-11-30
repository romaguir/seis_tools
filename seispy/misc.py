import obspy
import numpy as np
import matplotlib.pyplot as plt
from seis_tools.ses3d import rotation
from scipy.interpolate import griddata

try:
   from obspy.geodetics.base import gps2dist_azimuth
except ImportError:
    from obspy.core.util.geodetics.base import gps2DistAzimuth as gps2dist_azimuth

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

def km_per_deg_lon(latitude):
   '''
   returns how many km there are per degree of longitude, at a given latitude

   args:
       latitude: latitude in degrees
   '''
   latitude = np.radians(latitude)
   km_per_deg = (2*np.pi*6371.0*np.cos(latitude))/360.0
   return km_per_deg

def get_fresnel_width(wavelength,d,L):
    '''
    returns the width of the first Fresnel zone in km

    
    params:
    wavelength: wavelength of seismic energy (velocity * period) in km
    d: distance from source in km
    L: distance between source and receiver in km
    '''

    return 2*np.sqrt( (wavelength*d*(L-d)/L) )

def get_km_per_deg(radius):
    km_per_deg = (2*np.pi*radius) / 360.0
    return km_per_deg

def AR_to_CMT_focal(Mw,strike,dip,rake,**kwargs):
    '''
    takes the the moment magnitude, strike, dip, and rake of
    a focal mechanism, and returns the components of the moment
    tensor in CMT format

    inputs-
    Mw: moment magnitude
    strike: strike of the fault plane (degrees)
    dip: dip of the fault plane (degrees)
    rake: rake of the fault (degrees)

    kwargs:
    k = constant used in converstion between Mo and Mw (default = 9.1)

    output-
    Mo,M: Mo = seismic moment in Nm, M = tuple containing (Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)

    '''
    from numpy import sin, cos
    k = kwargs.get('k',9.1)

    #phi = np.radians(strike)
    #lamb = np.radians(rake)
    #delta = np.radians(dip)
    dip = np.radians(dip)
    rake = np.radians(rake)
    strike = np.radians(strike)

    Mo = 10**(1.5*Mw + k)

    #Mxx = -Mo*((sin(dip)*cos(rake)*sin(2.0*strike)) + 
    #           (sin(2.0*dip)*sin(rake)*(sin(strike)**2)))
    Mxx = -Mo*((sin(dip)*cos(rake)*sin(2.0*strike)) + 
               (sin(2.0*dip)*sin(rake)*(sin(strike)**2)))

    Mxy = Mo*((sin(dip)*cos(rake)*cos(2.0*strike)) + 
             (0.5*sin(2.0*dip)*sin(rake)*sin(2.0*strike)))

    Mxz = -Mo*((cos(dip)*cos(rake)*cos(strike)) + 
               (cos(2.0*dip)*sin(rake)*sin(strike)))

    Myy = Mo*((sin(dip)*cos(rake)*sin(2.0*strike)) - 
              (sin(2.0*dip)*sin(rake)*(cos(strike)**2)))

    Myz = -Mo*( (cos(dip)*cos(rake)*sin(strike)) - 
                (cos(2.0*dip)*sin(rake)*cos(strike)))

    Mzz = Mo*sin(2.0*dip)*sin(rake)

    Mrr = Mzz
    Mtt = Mxx
    Mpp = Myy
    Mrt = Mxz
    Mrp = -Myz
    Mtp = -Mxy

    #return Mo,(Mxx,Myy,Mzz,Mxy,Mxz,Myz)
    return Mo,(Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)

def write_CMTSOLUTION(M,evla,evlo,evdp,**kwargs):
    '''
    Takes the six independent elements of a moment tensor along with
    event location information, and writes a file for CMT file for
    use with specfem3d globe.

    inputs-
    M: [Mrr,Mtt,Mpp,Mrt,Mrp,Mtp] (moment tensor components in Nm)
    evla: earthquake latitude
    evlo: earthquake longitude
    evdp: earthquake depth (km)

    kwargs-
    file_name: name of the file to be written (default = 'CMTSOLUTION')
    event_name: name of the event
    mb: body wave magnitude (doesn't  matter in specfem simulations)
    ms: surface wave magnitude (doesn't matter in specfem simulations)
    time_shift: time shift in s (default = 0.0)
    half_dur: half duration in s (default = 0.0)
    date_time: origin time of event given in UTCDateTime

    '''
    from obspy import UTCDateTime

    file_name = kwargs.get('file_name','CMTSOLUTION')
    event_name = kwargs.get('event_name','XXX')
    region = kwargs.get('region','UNKNOWN')
    mb = kwargs.get('mb',1.0)
    ms = kwargs.get('ms',1.0)
    time_shift = kwargs.get('time_shift',0.0)
    half_dur = kwargs.get('half_dur',0.0)
    date_time = kwargs.get('date_time','2000-01-01T01:01:01.01')

    date_time = UTCDateTime(date_time)
    year = date_time.year
    month = date_time.month
    day = date_time.day
    hour = date_time.hour
    minute = date_time.minute
    second = date_time.second

    Mrr = M[0]
    Mtt = M[1]
    Mpp = M[2]
    Mrt = M[3]
    Mrp = M[4]
    Mtp = M[5]

    f = open(file_name,'w')
    f.write('PDE {} {} {} {} {} {} {} {} {} {} {} {}'.format(year,
        month,day,hour,minute,second,evla,evlo,evdp,mb,ms,region)+'\n')
    f.write('event name:     {}'.format(event_name)+'\n')
    f.write('time shift:     {}'.format(time_shift)+'\n')
    f.write('half duration:  {}'.format(half_dur)+'\n')
    f.write('latitude:       {}'.format(evla)+'\n')
    f.write('longitude:      {}'.format(evlo)+'\n')
    f.write('depth:          {}'.format(evdp)+'\n')
    f.write('Mrr:            {}'.format(Mrr)+'\n')
    f.write('Mtt:            {}'.format(Mtt)+'\n')
    f.write('Mpp:            {}'.format(Mpp)+'\n')
    f.write('Mrt:            {}'.format(Mrt)+'\n')
    f.write('Mrp:            {}'.format(Mrp)+'\n')
    f.write('Mtp:            {}'.format(Mtp)+'\n')
    f.close()

def write_STATIONS(st,**kwargs):
    '''
    takes a stream and writes a STATIONS file for use in specfem3d

    inputs-
    st: obspy stream

    **kwargs-
    file_name: name of the file to be written (default = 'STATIONS')
    '''
    file_name = kwargs.get('file_name','STATIONS')
    f = open(file_name,'w')

    for tr in st:

        station = tr.stats.station
        network = tr.stats.network 
        lat = tr.stats.sac['stla']
        lon = tr.stats.sac['stlo']
        elevation = tr.stats.sac['stel']
        depth = tr.stats.sac['stdp']

        f.write('{} {} {:4.2f} {:4.2f} {:4.2f} {:4.2f}'.format(station,network,lat,lon,elevation,depth)+'\n')

def find_NEX_XI(Tmin,Nchunks,angular_width=90.0):
    if Nchunks == 1:
        NEX_XI = (17.0 / Tmin) * (angular_width/90.0) * 256

    return NEX_XI

def fold_noise_correlation(tr):
    '''
    takes in a noise correlation function (as a trace object), and  
    folds it (i.e., averages the positive and negative lag times)
    '''

    l = len(tr.data)
    if l % 2 != 0:
        l -= 1

    npts = l/2

    half1 = tr.data[0:npts]
    half2 = tr.data[npts+1::]
    folded = half1[::-1] + half2

    tr_new = obspy.Trace()
    tr_new.stats = tr.stats
    tr_new.data = folded
    tr_new.stats.npts = npts

    return tr_new

def multiple_filter(tr,nbands,bandwidth,fmin,fmax,trim=True,**kwargs):
    '''
    not sure which bandwidth to use, but 0.05 seems to work best
    '''

    t_start = kwargs.get('t_start',50)
    time_len = kwargs.get('time_len',500)
    constant_period_band = kwargs.get('constant_period_band',False)
    virt_src = kwargs.get('virt_src','NA')
    rec_name = kwargs.get('rec_name','NA')
    #period_bandwidth = kwargs.get('period_bandwidth',20.0)

    tr.trim(tr.stats.starttime+t_start,tr.stats.starttime+time_len)
    #tr.normalize()

    if constant_period_band:
        T_center = np.linspace(1./fmax,1./fmin,nbands)
        f_center = 1./T_center
    else:
        f_center = np.linspace(fmin,fmax,nbands)

    x_s = []
    y_s = []
    z_s = []
    max_val = []

    try:
       dist = tr.stats.sac['dist']
    except KeyError:
       #dist = tr.stats.sac['gcarc']*111.19
       distaz = gps2dist_azimuth(tr.stats.sac['evla'],tr.stats.sac['evlo'],
               tr.stats.sac['stla'],tr.stats.sac['stlo'])
       dist = distaz[0]/1000.

    samprate = tr.stats.sampling_rate
    for f in f_center:
        tr_new = tr.copy()
        freqmin = f-(bandwidth/2.0)
        freqmax = f+(bandwidth/2.0)

        if freqmin < 0.002:
           freqmin = 0.002
           print 'jollygood!'

        print 'Freqs:', freqmin,freqmax, 'Periods:', 1./freqmax, 1./freqmin

        if freqmin > 0.0:
           #tr_new.filter('bandpass',freqmin=f-(bandwidth/2.0),freqmax=f+(bandwidth/2.0),
           #        corners=4,zerophase=True)
           tr_new.filter('bandpass',freqmin=freqmin,freqmax=freqmax,
                   corners=4,zerophase=True)
        else:
           tr_new.filter('lowpass',freq=freqmax)

        data_envelope = obspy.signal.filter.envelope(tr_new.data)

        #plt.plot(data_envelope)
        #plt.plot(tr_new.data)
        #plt.show()

        #data_envelope /= np.max(data_envelope)

        t = np.arange(t_start, tr.stats.npts / samprate, 1 / samprate)
        veloc = dist/t
        x = np.zeros(len(t))
        x[:] = 1./f

        #plt.plot(t,data_envelope,'r')
        #plt.plot(t,tr_new.data,'k')
        #plt.show()

        for i in range(0,len(x)):
            x_s.append(x[i])
            y_s.append(veloc[i])
            z_s.append(data_envelope[i])

        max_i = np.argmax(data_envelope)
        max_val.append(veloc[max_i])
        #max_val.append(np.max(data_envelope))

    points = (x_s,y_s)
    data = z_s
    x_axis = np.linspace(5,40,100)
    y_axis = np.linspace(2.5,4.0,100)
    grid_x,grid_y = np.meshgrid(x_axis,y_axis)
    grid_z = griddata(points, data, (grid_x,grid_y))

    #plt.scatter(x_s,y_s,c=z_s,edgecolor='none',s=50)
    #plt.scatter(x_s,y_s,c=z_s,edgecolor='none',s=50,vmin=0.0,vmax=0.3)

    plt.pcolormesh(x_axis,y_axis,grid_z)
    plt.xlabel('period (s)')
    plt.ylabel('velocity (km/s)')
    plt.colorbar()
    #plt.scatter(1./f_center,max_val,marker='*',s=75,c='w')


    #group dispersion maximum
    plt.scatter(1./f_center,max_val,marker='*',s=75,c='w')
    plt.xlim([5,40])
    plt.ylim([2.5,4.0])

    print len(max_val)
    print len(x_s)
    print len(y_s)
    plt.title(virt_src+' '+rec_name)
    plt.savefig('{}_{}_groupveloc.pdf'.format(virt_src,rec_name),format='pdf')
    plt.show()
