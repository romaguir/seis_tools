import obspy
import pandas
from obspy.taup import TauPyModel
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import Catalog
try:
    from obspy.geodetics import gps2dist_azimuth
except ImportError:
    from obspy.core.util.geodetics import gps2DistAzimuth as gps2dist_azimuth
try:
   from obspy.geodetics import kilometer2degrees
except ImportError:
   from obspy.core.util.geodetics import kilometer2degrees
#-------------------------------------------------------------------------------------

def data_downloader(save_directory,**kwargs):
   '''
   This is a function that uses obspy to download lots of data, and save it to a
   stream.

   params:
   save_directory - path to save directory [dtype = str]

   **kwargs:
   lat0 - latitude center of geographic search           [dtype = float]
   lon0 - longitude center of geographic search          [dtype = float]
   minrad - minimum distance from (lat0,lon0) in degrees [dtype = float]
   minmag - minimum earthquake magnitude                 [dtype = float]
   maxmag - maximum earthquake magnitude                 [dtype = float]
   starttime - beginning time of search       [dtype = obspy UTCDateTime instance]
   endtime - end time of search               [dtype = obspy UTCDateTime instance]
   network - name of seismic network (can be *)           [dtype = str]
   min_before_P - minutes before P wave arrival          [dtype = float]
   min_after_P - minutes before P wave arrival           [dtype = float]
   save_format - format to save data (only pickle so far) [dtype = str]
   gcarc_max - only download data from stations closer than gcarc_max (degrees) [dtype = float]
   gcarc_min - only download data from stations farther than gcarc_max (degrees) [dtype = float]
   '''

   #------------------------------------------------------------------------------
   #initializations
   #------------------------------------------------------------------------------
   model = TauPyModel('ak135')
   client = Client('IRIS',timeout=300)
   event_num = 1
   
   #------------------------------------------------------------------------------
   #get all earthquakes between 2004 and 2013 matching the criteria
   #------------------------------------------------------------------------------
   lat0 = kwargs.get('lat0',40.0)
   lon0 = kwargs.get('lon0',-100.0)
   minmag = kwargs.get('minmag',7.9)
   maxmag = kwargs.get('maxmag',8.0)
   minrad = kwargs.get('minrad',35.0)
   maxrad = kwargs.get('maxrad',80.0)
   starttime = kwargs.get('starttime',UTCDateTime('2004-01-01'))
   endtime   = kwargs.get('endtime',UTCDateTime('2014-01-01'))
   channels  = kwargs.get('channels','all')
   network   = kwargs.get('network','TA')
   min_before_P = kwargs.get('min_before_P',1.0)
   min_after_P  = kwargs.get('min_before_P',5.0)
   save_format  = kwargs.get('save_format','PICKLE')
   gcarc_max = kwargs.get('gcarc_max',95.0)
   gcarc_min = kwargs.get('gcarc_min',0.0)

   cat = client.get_events(starttime=starttime,
                           endtime=endtime,
                           minmagnitude=minmag,
                           maxmagnitude=maxmag)
   
   #------------------------------------------------------------------------------
   #filter the catalog in a specified range
   #------------------------------------------------------------------------------
   temp_events = []
   for ev in cat:
      ev_lat = ev.origins[0]['latitude']
      ev_lon = ev.origins[0]['longitude']
      distaz = gps2dist_azimuth(ev_lat,ev_lon,lat0,lon0)
      dist_m = distaz[0]
      dist_deg = kilometer2degrees((dist_m/1000.0))
      if dist_deg >= minrad and dist_deg <= maxrad:
         temp_events.append(ev)
   
   cat = Catalog(events=temp_events)
   cat.plot()
   
   #------------------------------------------------------------------------------
   #get waveforms
   #------------------------------------------------------------------------------

   #loop through events----------------------------------------------------------
   for ev in cat:
      origin = ev.origins
      origintime = origin[0]['time']
      evlo       = origin[0]['longitude']
      evla       = origin[0]['latitude']
      evdp       = origin[0]['depth']/1000.0
      inventory  = client.get_stations(network='TA',
                                       starttime = origintime,
                                       endtime = origintime+60.0)
      station_list =  inventory[0]
      location_string = ev.event_descriptions[0].text.replace(' ','_')
      magnitude_string = '_M'+str(ev.magnitudes[0].mag)+'_'
      date_string = str(origintime.year)+'-'+str(origintime.month)+'-'+str(origintime.day)
      event_output_name = location_string+magnitude_string+date_string
      k = 0
    
      #some output---------------------------------------------------------------
      print '-------------------------------------------------------------------'
      print 'Event '+str(event_num),"of ", cat.count()
      print 'Start time = ',UTCDateTime.utcnow()
      print "Downloading data for event ",event_output_name
      for network in inventory:
         netw_code         = network.code
         stations_selected = network.selected_number_of_stations
         print "Found ", stations_selected, " stations in network ", netw_code
      print '\n'
      event_num += 1

      #setup lists for bulk download---------------------------------------------
      sac_dict_list = []
      bulk_list_BHE = []
      bulk_list_BHN = []
      bulk_list_BHZ = []

      #range filter stations----------------------------------------------------
      #TODO

      #loop through stations-----------------------------------------------------
      for station in station_list:

         #get station statistics-------------------------------------------------
         stla = station.latitude
         stlo = station.longitude
         distaz   = gps2dist_azimuth(evla,evlo,stla,stlo)  
         dist_m   = distaz[0]
         dist_km  = distaz[0]/1000.0
         dist_deg = kilometer2degrees((dist_m/1000.0))
         az       = distaz[1]
         baz      = distaz[2]

         #get predicted time of P arrival with taup-------------------------------
         arr = model.get_travel_times(source_depth_in_km = evdp,
                                      distance_in_degree = dist_deg,
                                      phase_list=['P','Pdiff'])
         P_time = arr[0].time
         P_arr = origintime + P_time
         t1 = P_arr - (60.0*min_before_P)   
         t2 = P_arr + (60.0*min_after_P)   
         o  = (60.0*min_before_P)-P_time

         #create list with sac information---------------------------------------
         sac_dict = {'dist':dist_km,'az':az,'baz':baz,'gcarc':dist_deg,'stlo':stlo,
                     'stla':stla,'evla':evla,'evlo':evlo,'evdp':evdp,'o':o}
         sac_dict_list.append(sac_dict)

         #create list for bulk waveform downloader--------------------------------
         st_info = inventory.get_contents()['stations']
         _network = st_info[k].split()[0].split('.')[0]
         _station = st_info[k].split()[0].split('.')[1]
         
         bulk_line_BHE = (_network,_station,'','BHE',t1,t2)
         bulk_list_BHE.append(bulk_line_BHE)
         bulk_line_BHN = (_network,_station,'','BHN',t1,t2)
         bulk_list_BHN.append(bulk_line_BHN)
         bulk_line_BHZ = (_network,_station,'','BHZ',t1,t2)
         bulk_list_BHZ.append(bulk_line_BHZ)

         #if sac_dict['gcarc'] >= 90.0:
            #print sac_dict

         k += 1
          
      #use bulk downloader to get stream containing all waveforms for an event---
      ste = client.get_waveforms_bulk(bulk_list_BHE,attach_response=True) 
      stn = client.get_waveforms_bulk(bulk_list_BHN,attach_response=True) 
      stz = client.get_waveforms_bulk(bulk_list_BHZ,attach_response=True) 

      #attach a sac dictionary to traces----------------------------------------- 
      for i in range(0,len(ste)):
         for j in range(0,len(bulk_list_BHE)):
            if bulk_list_BHE[j][1] == ste[i].stats.station:
               ste[i].stats.sac = {}
               ste[i].stats.sac['dist'] = sac_dict_list[j]['dist']
               ste[i].stats.sac['az'] = sac_dict_list[j]['az']
               ste[i].stats.sac['baz'] = sac_dict_list[j]['baz']
               ste[i].stats.sac['gcarc'] = sac_dict_list[j]['gcarc']
               ste[i].stats.sac['stla'] = sac_dict_list[j]['stla']
               ste[i].stats.sac['stlo'] = sac_dict_list[j]['stlo']
               ste[i].stats.sac['evla'] = sac_dict_list[j]['evla']
               ste[i].stats.sac['evlo'] = sac_dict_list[j]['evlo']
               ste[i].stats.sac['evdp'] = sac_dict_list[j]['evdp']
               ste[i].stats.sac['o'] = sac_dict_list[j]['o']

      for i in range(0,len(stn)):
         for j in range(0,len(bulk_list_BHN)):
            if bulk_list_BHN[j][1] == stn[i].stats.station:
               stn[i].stats.sac = {}
               stn[i].stats.sac['dist'] = sac_dict_list[j]['dist']
               stn[i].stats.sac['az'] = sac_dict_list[j]['az']
               stn[i].stats.sac['baz'] = sac_dict_list[j]['baz']
               stn[i].stats.sac['gcarc'] = sac_dict_list[j]['gcarc']
               stn[i].stats.sac['stla'] = sac_dict_list[j]['stla']
               stn[i].stats.sac['stlo'] = sac_dict_list[j]['stlo']
               stn[i].stats.sac['evla'] = sac_dict_list[j]['evla']
               stn[i].stats.sac['evlo'] = sac_dict_list[j]['evlo']
               stn[i].stats.sac['evdp'] = sac_dict_list[j]['evdp']
               stn[i].stats.sac['o'] = sac_dict_list[j]['o']
      
      for i in range(0,len(stz)):
         for j in range(0,len(bulk_list_BHZ)):
            if bulk_list_BHE[j][1] == stz[i].stats.station:
               stz[i].stats.sac = {}
               stz[i].stats.sac['dist'] = sac_dict_list[j]['dist']
               stz[i].stats.sac['az'] = sac_dict_list[j]['az']
               stz[i].stats.sac['baz'] = sac_dict_list[j]['baz']
               stz[i].stats.sac['gcarc'] = sac_dict_list[j]['gcarc']
               stz[i].stats.sac['stla'] = sac_dict_list[j]['stla']
               stz[i].stats.sac['stlo'] = sac_dict_list[j]['stlo']
               stz[i].stats.sac['evla'] = sac_dict_list[j]['evla']
               stz[i].stats.sac['evlo'] = sac_dict_list[j]['evlo']
               stz[i].stats.sac['evdp'] = sac_dict_list[j]['evdp']
               stz[i].stats.sac['o'] = sac_dict_list[j]['o']

      #remove instrument response---------------------------------------------------
      print "Removing instrument response"
      ste.remove_response()
      stn.remove_response()
      stz.remove_response()

      #write streams----------------------------------------------------------------
      print "Writing stream to output file"
      ste.write(filename=save_directory+'/'+event_output_name+'_BHE.pk',format='PICKLE')
      stn.write(filename=save_directory+'/'+event_output_name+'_BHN.pk',format='PICKLE')
      stz.write(filename=save_directory+'/'+event_output_name+'_BHZ.pk',format='PICKLE')

def filter_catalog(catalog,lon0,lat0,minrad,maxrad):
   temp_events = []
   for ev in catalog:
      ev_lat = ev.origins[0]['latitude']
      ev_lon = ev.origins[0]['longitude']
      distaz = gps2dist_azimuth(ev_lat,ev_lon,lat0,lon0)
      dist_m = distaz[0]
      dist_deg = kilometer2degrees((dist_m/1000.0))
      if dist_deg >= minrad and dist_deg <= maxrad:
         temp_events.append(ev)
   cat = Catalog(events=temp_events)
   return cat

