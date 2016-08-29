'''
seis_plot.py includes all functions for plotting seismic data
'''

import matplotlib
import scipy
import obspy
import h5py
import numpy as np

#from seispy.data import phase_window
from data import phase_window
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
import obspy.signal.filter
import obspy.signal
from matplotlib import colors, ticker, cm
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import os
from scipy.optimize import curve_fit
import math
from matplotlib.colors import LightSource
from mpl_toolkits.basemap import Basemap
from cycler import cycler
import multiprocessing
import shutil
model = TauPyModel(model="ak135")

def precursor_PKIKP(seis_stream_in,precursor_onset,time_offset,name=False):
    '''
    Produce colorbar of PKPdf scatterers from Mancinelli & Shearer 2011
    PARAMETERS
    __________

    seis_stream_in: obspy.core.stream.Stream object.

    precursor_onset = h5py file

    time_offset = integer/float amount of seconds to shift the arrival
                 to match PKPdf on zero

    name = default = False. If a string is passed, it will save the
                 image to pdf in the
           directory work4/IMAGES
    '''
    event_depth = seis_stream_in[0].stats.xh['source_depth_in_km']

    f = h5py.File(precursor_onset,'r')

    #Find reciever latitude and correct for coordinate system
    def receiver_latitude(tr):
        station = tr.stats.station.split('_')
        lat = float(station[0]+'.'+station[1])
        return lat

    seis_stream = seis_stream_in.copy()
    shearer_stream = obspy.core.stream.Stream()
    envelope_dict = {}
    waveform_dict = {}

    t = seis_stream[0].stats.starttime
    samp_rate = seis_stream[0].stats.sampling_rate

    for tr in seis_stream:
        lat = receiver_latitude(tr)
        if 120. <= lat <= 145.:
            tr.differentiate()
            tr.filter('bandpass',freqmin=0.3,freqmax=2.5)
            arrivals = model.get_travel_times(source_depth_in_km=event_depth,
                       distance_in_degree = lat, phase_list = ['PKiKP'])
            PKiKP = arrivals[0].time+time_offset
            array = tr.slice(t+PKiKP-30,t+PKiKP+20)
            array_norm = tr.slice(t+PKiKP-18,t+PKiKP-15).data.max()
            data_envelope = obspy.signal.filter.envelope(array.data)
            data_waveform = (array.data/array_norm).clip(min=0)*2e-2
            #data_waveform = (array.data/array_norm)*2e-1
            envelope_dict[lat] = data_envelope
            waveform_dict[lat] = data_waveform

    rows = envelope_dict[120].shape[0]
    cols = len(envelope_dict.keys())
    waveform_array = np.zeros((rows,cols))
    envelope_array = np.zeros((rows,cols))

    for idx, keys in enumerate(sorted(envelope_dict)):
        length = envelope_dict[keys][0:rows].size
        envelope_array[0:length,idx] = envelope_dict[keys][0:rows]
        waveform_array[0:length,idx] = waveform_dict[keys][0:rows]

    waveform_array = np.fliplr(np.transpose(np.flipud(waveform_array)))
    precursor_onset = int(waveform_array.shape[1]*3./5.)

    # Adjust precursor amplitude
    waveform_array[:,0:precursor_onset] *= 1
    time = np.linspace(-25,25,num=waveform_array.shape[1])

    fig, (ax,ax1) = plt.subplots(1,2,figsize=plt.figaspect(0.5))

    # Waveform axis
    for ii in range(0,waveform_array.shape[0])[::-1]:
        waveform_array[ii,:] += (120+(ii/10.))
        base = (120+(ii/10.))*np.ones(waveform_array[ii,:].shape)
        #ax.plot(time,waveform_array[ii,:],c='k',lw=0)
        #ax.plot(time,base,c='w',alpha=0.0)
        ax.fill_between(time, waveform_array[ii,:], base,
                where=waveform_array[ii,:] >= base,
                facecolor='#E6A817', interpolate=True,lw=0.6,alpha=0.8)
        ax.fill_between(time, waveform_array[ii,:], base,
                where=waveform_array[ii,:] <= base,
                facecolor='blue', interpolate=True,lw=0.6,alpha=0.8)

    x = np.linspace(120,145,num=f['2891'][...].shape[0])
    for items in f:
       onset = f[items][...]
       onset[onset > 0] = np.nan
       ax.plot(onset,x,color='k',lw=2)
       ax1.plot(x,onset,color='w',lw=2)

    ax.set_ylim([120,148])
    ax.set_xlim([-25,25])
    ax.set_ylabel('Range (degrees)',fontsize=14)
    ax.set_xlabel(r'Seconds after $PKPdf$',fontsize=14)
    #ax.plot(y,x,color='k',lw=2)

    # Colorbar axis
    envelope_array = envelope_array/envelope_array.max()
    image = ax1.imshow(np.log10(np.flipud(envelope_array)+1e-8),aspect='auto',
            cmap='Spectral_r',extent=[120,145,-25,25],
            interpolation='none',vmin=-3.5,vmax=-2.5)
    ax1.set_xlabel('Range (degrees)',fontsize=14)
    ax1.set_ylabel(r'Seconds after $PKPdf$',fontsize=14)
    ax1.set_ylim(-25,25)
    ax1.set_xlim(120,145)
    ax1.text(121,-19,'CMB',color='w',fontsize=14)
    #ax1.plot(x,y,color='w',lw=2)
    cbar = fig.colorbar(image,ax=ax1)
    cbar.solids.set_rasterized(True)
    cbar.set_label(r'$\log_{10}(Amplitude)$',fontsize=14)

    return waveform_array
    #if name == False:
    #    plt.show()
    #else:
    #    fig.suptitle(str(name),fontsize=16)
    #    plt.savefig('/home/samhaug/Documents/Figures/'+name+'.pdf')

def vespagram(st_in,**kwargs):
    '''
    make vespagram of stream object. Recommended to pass stream object
    through remove_dirty
    '''

    window_tuple = kwargs.get('window_tuple',(-10,230))
    window_phase = kwargs.get('window_phase',['P'])
    phase_list = kwargs.get('phase_list',False)
    plot_line = kwargs.get('plot_line',False)
    n_root = kwargs.get('n_root',2.0)
    save = kwargs.get('save',False)

    st = obspy.core.Stream()
    for idx, tr in enumerate(st_in):
        st += phase_window(tr,window_phase,window_tuple)

    def mean_range(st):
        a = []
        for tr in st:
            a.append(tr.stats.sac['gcarc'])
        mn_r = np.mean(a)
        return mn_r

    def roll_zero(array,n):
        if n < 0:
            array = np.roll(array,n)
            array[n::] = 0
        else:
            array = np.roll(array,n)
            array[0:n] = 0
        return array

    def slant_stack(st,mn_range,slowness,n):
        d = st[0].stats.delta
        R = np.zeros(st[0].data.shape[0])
        for tr in st:
            az = tr.stats.sac['gcarc']-mn_range
            shift_in_sec = slowness*az
            shift_in_bin = int(shift_in_sec/d)
            x = roll_zero(tr.data,shift_in_bin)
            R += np.sign(x)*pow(np.abs(x),1.0/n)
        R = R/float(len(st))
        yi = R*pow(abs(R),n-1)
        hil = scipy.fftpack.hilbert(yi)
        yi = pow(hil**2+R**2,1/2.)
        return yi,R*(3.)

    def phase_plot(ax,evdp,degree,phases,text_color):
        P_arrivals = model.get_travel_times(distance_in_degree = degree,
        source_depth_in_km = evdp,
        phase_list = ['P'])

        P_slowness = P_arrivals[0].ray_param_sec_degree
        P_time = P_arrivals[0].time

        arrivals = model.get_travel_times(distance_in_degree=degree,
        source_depth_in_km=evdp,
        phase_list = phases)
        if len(arrivals) != 0:
            colors = ['b','g','r','c','m','y','k']
            for idx, ii in enumerate(arrivals):
                p = ii.ray_param_sec_degree-P_slowness
                time = ii.time-P_time
                ax.scatter(time,p,s=22,marker='D',c='w',zorder=20)
                #ax.text(time,p,ii.name,fontsize=16,color=text_color)


    mn_r = mean_range(st)
    evdp = st[0].stats.sac['evdp']

    st.normalize()

    vesp_y = np.linspace(0,0,num=st[0].data.shape[0])
    vesp_R = np.linspace(0,0,num=st[0].data.shape[0])
    for ii in np.arange(-1.5,1.5,0.1):
        yi,R = slant_stack(st,mn_r,ii,n_root)
        vesp_y= np.vstack((vesp_y,yi))
        vesp_R= np.vstack((vesp_R,R))
    vesp_y = vesp_y[1::,:]
    vesp_R = vesp_R[1::,:]
    vesp_y = vesp_y/ vesp_y.max()

    fig, ax = plt.subplots(2,sharex=True,figsize=(15,10))

    image_0 = ax[0].imshow(np.log(vesp_y),aspect='auto',interpolation='lanczos',
            extent=[window_tuple[0],window_tuple[1],-1.5,1.5],
            cmap='inferno',vmin=-6,vmax=-2)

    vesp_y = np.linspace(0,0,num=st[0].data.shape[0])
    vesp_R = np.linspace(0,0,num=st[0].data.shape[0])
    for ii in np.arange(-1.5,1.5,0.1):
        yi,R = slant_stack(st,mn_r,ii,1.0)
        vesp_y= np.vstack((vesp_y,yi))
        vesp_R= np.vstack((vesp_R,R))
    vesp_y = vesp_y[1::,:]
    vesp_R = vesp_R[1::,:]
    vesp_y = vesp_y/ vesp_y.max()

    image_1 = ax[1].imshow(np.log(vesp_y), aspect='auto',
            interpolation='lanczos', extent=[window_tuple[0],
            window_tuple[1],-1.5,1.5],cmap='inferno', vmin=-6,vmax=-2)

    cbar_0 = fig.colorbar(image_0,ax=ax[0])
    cbar_0.set_label('Log(Normalized seismic energy)')
    cbar_1 = fig.colorbar(image_1,ax=ax[1])
    cbar_1.set_label('Log(Normalized seismic energy)')
    ax[0].set_xlim(window_tuple)
    ax[1].set_xlim(window_tuple)
    ax[1].xaxis.set(ticks=range(window_tuple[0],window_tuple[1],10))
    ax[0].xaxis.set(ticks=range(window_tuple[0],window_tuple[1],10))
    ax[0].set_ylim([-1.5,1.5])
    ax[1].set_ylim([-1.5,1.5])
    ax[0].grid(color='w',lw=2)
    ax[1].grid(color='w',lw=2)
    ax[1].set_ylabel('Slowness (s/deg)')
    ax[0].set_ylabel('Slowness (s/deg)')
    ax[1].set_xlabel('Seconds after {}'.format(window_phase[0]))
    ax[0].set_title('Start: {} \n Source Depth: {} km, Ref_dist: {} deg, {} \
                     \n Top : N-root = {} Bottom: N-root = 1'
                  .format(st[0].stats.starttime,
                  round(st[0].stats.sac['evdp'],3),
                  round(mn_r,3), os.getcwd().split('-')[3],
                  str(n_root)))

    if phase_list:
       phase_plot(ax[0],evdp,mn_r,phase_list,text_color='green')
       phase_plot(ax[1],evdp,mn_r,phase_list,text_color='green')
       #phase_plot(axR,evdp,mn_r,phase_list,text_color='black')

    if save != False:
        plt.savefig(save+'/vespagram.pdf',format='pdf')

    time_vec = np.linspace(window_tuple[0], window_tuple[1],
               num=vesp_R.shape[1])

    figR, axR= plt.subplots(1,figsize=(14,7))

    for idx, ii in enumerate(np.arange(1.5,-1.5,-0.1)):
        vesp_R[idx,:] += ii
        axR.fill_between(time_vec,ii,vesp_R[idx,:],where=vesp_R[idx,:] >= ii,
                         facecolor='goldenrod',alpha=0.5,lw=0.5)
        axR.fill_between(time_vec,ii,vesp_R[idx,:],where=vesp_R[idx,:] <= ii,
                         facecolor='blue',alpha=0.5,lw=0.5)

    axR.set_xlim(window_tuple)
    axR.set_ylim((-1.6,1.6))
    axR.set_ylabel('Slowness (s/deg)')
    axR.set_xlabel('Seconds after P')
    axR.set_title('Start: {} \n Source Depth: {} km, Ref_dist: {} deg, {}'
                  .format(st[0].stats.starttime,
                   round(st[0].stats.sac['evdp'],3),round(mn_r,3),
                   os.getcwd().split('-')[3]))

    if save != False:
        plt.savefig(save+'/wave.pdf',format='pdf')
    else:
        plt.show()

def plot(tr,**kwargs):
    '''
    plot trace object
    phase = optional list of phases. list of strings corresponding
            to taup names
    '''
    phases = kwargs.get('phase_list',False)
    window = kwargs.get('window',False)

    fig, ax = plt.subplots(figsize=(23,9))
    if phases != False:
        arrivals = model.get_travel_times(
                   distance_in_degree=tr.stats.sac['gcarc'],
                   source_depth_in_km=tr.stats.sac['evdp'],
                   phase_list = phases)
        window_list = []
        colors = ['b','g','r','c','m','y','k']
        if len(arrivals) != 0:
            for idx, ii in enumerate(arrivals):
                ax.axvline(ii.time,label=ii.name,c=colors[idx])
                window_list.append(ii.time)
    ax.legend()

    time = np.linspace(-1*tr.stats.sac['o'],
           (tr.stats.delta*tr.stats.npts)-tr.stats.sac['o'],
           num=tr.stats.npts)
    ax.plot(time,tr.data,c='k')
    ax.grid()
    ax.set_title('Network: {}, Station: {}, Channel: {},\
 Dist (deg): {}, Depth (km): {} \nStart: {} \nEnd: {}'.format(
                  tr.stats.network,
                  tr.stats.station,
                  tr.stats.channel,
                  round(tr.stats.sac['gcarc'],3),
                  round(tr.stats.sac['evdp'],3),
                  tr.stats.starttime,
                  tr.stats.endtime))
    ax.set_xlabel('Time (s), sampling_rate: {}, npts: {}'
                 .format(tr.stats.sampling_rate,tr.stats.npts))

    if window == True:
        ax.set_xlim([min(window_list)-300,max(window_list)+300])

    plt.show()

def parallel_add_to_axes(trace_tuple):
    '''
    parallel plotting function to be used by section
    '''
    data = trace_tuple[0]
    time = trace_tuple[1]
    dist = trace_tuple[2]
    line = matplotlib.lines.Line2D(time,data+dist,alpha=0.5,c='k',lw=1)
    return line

def section(st,**kwargs):
    '''
    Plot record section of obspy stream object

    kwargs:
    shift: True or False. set start of every trace to be t=0
    flip_axes: True or False. if True, plots distance on x axis and time on y
    '''
    labels = kwargs.get('labels',False)
    phases = kwargs.get('phase_list',False)
    fill = kwargs.get('fill',False)
    shift = kwargs.get('shift',False)
    save = kwargs.get('save',False)
    color = kwargs.get('color','k')
    showplot = kwargs.get('showplot',True)
    axis = kwargs.get('axis','none')
    flip_axes = kwargs.get('flip_axes',False)

    def main():
        p_list,name_list,dist_list = p_list_maker(st)
        lim_tuple = ax_limits(p_list)

        if axis == 'none':
           fig, ax = plt.subplots(figsize =(10,15))
        else:
           ax = axis

        for ii in p_list:
            add_to_axes(ii,ax)

        if phases != False:
            phase_plot(lim_tuple,50.,st[0].stats.sac['evdp'],phases,ax)

        if flip_axes == False:
           ax.set_ylabel('Distance (deg)')
           ax.set_xlabel('Seconds After Event')
        elif flip_axes == True:
           ax.set_xlabel('Distance (deg)')
           ax.set_ylabel('Seconds After Event')

        if shift:
            if flip_axes == False:
               ax.set_xlabel('Seconds')
            elif flip_axes == True:
               ax.set_ylabel('Seconds')

        if labels:
            y1, y2 = ax.get_ylim()
            ax_n = ax.twinx()
            ax_n.set_yticks(dist_list)
            ax_n.set_yticklabels(name_list)
            ax_n.set_ylim(y1,y2)

        if save != False:
            plt.savefig(save+'/section.pdf',format='pdf')

        if showplot == True:
           plt.show()
        else:
           print "this should return an axis object"

        return ax

    def phase_plot(lim_tuple,ref_degree,evdp,phases,ax):
        arrivals = model.get_travel_times(distance_in_degree=ref_degree,
        source_depth_in_km=evdp,
        phase_list = phases)
        print "arrivals", arrivals
        if len(arrivals) != 0:
            colors = ['b','g','r','c','m','y','k']
            for idx, ii in enumerate(arrivals):
                p = ii.ray_param_sec_degree
                time = ii.time
                x = np.linspace(time-1000,time+1000)
                y = (1/p)*(x-time)+ref_degree
                ax.plot(x,y,alpha=0.3,label=ii.name,c=colors[idx])
                ax.legend()

    def plotter(art,lim_tuple,ax):
        ax.grid()
        ax.set_xlim(lim_tuple[0])
        ax.set_ylim(lim_tuple[1])
        for ii in art:
            ax.add_line(ii)

    def add_to_axes(trace_tuple,ax):
        data = trace_tuple[0]
        time = trace_tuple[1]
        dist = trace_tuple[2]
        if flip_axes == False:
           ax.plot(time,data+dist,alpha=0.5,c=color,lw=1)
        elif flip_axes == True:
           ax.plot(data+dist,time,alpha=0.5,c=color,lw=1)
      

        if fill:
            ax.fill_between(time, dist, data+dist, where=data+dist <= dist,
                            facecolor='k', alpha=0.5, interpolate=True)

    def p_list_maker(st):
        p_list = []
        name_list = []
        dist_list = []
        for tr in st:
            data = tr.data
            o = tr.stats.sac['o']
            if shift:
                o = 0
            time = np.linspace(-1*o,(tr.stats.delta*tr.stats.npts)-o,
                   num=tr.stats.npts)
            dist = tr.stats.sac['gcarc']
            name_list.append(str(tr.stats.network)+'.'+str(tr.stats.station))
            dist_list.append(dist)
            p_list.append((data,time,dist))
        return p_list,name_list,dist_list

    def ax_limits(p_list):
        range_list = []
        for ii in p_list:
            range_list.append([ii[2],ii[1][0],ii[1][-1]])
        range_list = np.array(range_list)
        min_range = min(range_list[:,0])
        max_range = max(range_list[:,0])
        min_time = min(range_list[:,1])
        max_time = max(range_list[:,2])
        return ([min_time,max_time],[min_range-3,max_range+3])

    m = main()
    return m

def fft(tr, freqmin=0.0, freqmax=2.0):
    '''
    plot fast fourier transform of trace object
    '''

    Fs = tr.stats.sampling_rate  # sampling rate
    Ts = tr.stats.delta # sampling interval
    time = tr.stats.endtime - tr.stats.starttime
    t = np.arange(0,time,Ts) # time vector

    n = len(tr.data) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(tr.data)/n # fft computing and normalization
    Y = Y[range(n/2)]
    if tr.data.shape[0] - t.shape[0] == 1:
        t = np.hstack((t,0))

    fig, ax = plt.subplots(2, 1,figsize=(15,8))
    ax[0].plot(t,tr.data,'k')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[0].grid()
    ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
    ax[1].set_xlim([freqmin,freqmax])
    ax[1].set_xticks(np.arange(freqmin,freqmax,0.25))
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    ax[1].grid()
    plt.show()

def express_plot(st, **kwargs):
    phase_list = kwargs.pop('phase_list',['P'])
    name = os.getcwd().split('/')[-1]
    fig_dir = '/home/samhaug/work1/Figures/'
    if os.path.exists(fig_dir+name):
        #os.rmdir(fig_dir+name)
        shutil.rmtree(fig_dir+name)
    os.mkdir(fig_dir+name)

    #vespagram(st,phase_list=phase_list,save=fig_dir+name)
    vespagram(st,phase_list=['P','pP','sP','S660P','S860P','S1260P','S1060P',
                          'S1460P','S1660P'],window_tuple=(-10,200),
                           save=fig_dir+name)
    section(st,shift=True,save=fig_dir+name)
    mapplot.source_reciever_plot(st,save=fig_dir+name)
    st.write(fig_dir+name+'/'+name+'.SAC',format='SAC')

#########################################################################
# Plot map of earthquake and station for a trace
#########################################################################
def map_trace(tr,ax='None',showpath=True,showplot=True,Lat_0=0.0,Lon_0=60.0):
   from mpl_toolkits.basemap import Basemap

   if ax == 'None':
       m = Basemap(projection='ortho',lat_0=Lat_0,lon_0=Lon_0,resolution='l')
       m.drawmapboundary()
       m.drawcoastlines()
       m.fillcontinents(color='gray',lake_color='white')
   else:
      m = ax

   x1,y1 = m(tr.stats.sac['evlo'],tr.stats.sac['evla'])
   x2,y2 = m(tr.stats.sac['stlo'],tr.stats.sac['stla'])
   m.scatter(x1,y1,s=200.0,marker='*',facecolors='y',edgecolors='k',zorder=99)
   m.scatter(x2,y2,s=20.0,marker='^',color='b',zorder=99)

   if showpath == True:
      m.drawgreatcircle(tr.stats.sac['evlo'],tr.stats.sac['evla'],
                        tr.stats.sac['stlo'],tr.stats.sac['stla'],
                        linewidth=1,color='k',alpha=0.5)

   if showplot == True:
      plt.show()
   else:
      return m

#########################################################################
# Plot map of earthquake and station for all traces in stream
#########################################################################
def map_stream(st,showpath=True,showplot=True,Lat_0=0.0,Lon_0=60.0):
   from mpl_toolkits.basemap import Basemap

   fig = plt.figure()
   plt.ion()
   m = Basemap(projection='ortho',lat_0=Lat_0,lon_0=Lon_0,resolution='l')
   m.drawmapboundary()
   m.drawcoastlines()
   m.fillcontinents(color='gray',lake_color='white')
   gc_lines = []

   for tr in st:
      x1,y1 = m(tr.stats.sac['evlo'],tr.stats.sac['evla'])
      x2,y2 = m(tr.stats.sac['stlo'],tr.stats.sac['stla'])
      m.scatter(x1,y1,s=200.0,marker='*',facecolors='y',edgecolors='k',zorder=99)
      station_pt = m.scatter(x2,y2,s=20.0,marker='^',color='b',zorder=99,picker=1)
      station_pt.set_label(tr.stats.station)

      if showpath == True:
         gc_line = m.drawgreatcircle(tr.stats.sac['evlo'],tr.stats.sac['evla'],
                                     tr.stats.sac['stlo'],tr.stats.sac['stla'],
                                     linewidth=1,color='k',alpha=0.5)
         gc_line[0].set_label(tr.stats.station)
         gc_lines.append(gc_line)

   def onpick(event):
      art = event.artist
      picked =  art.get_label()
      print "removing station(s) ", picked
      st_to_remove = st.select(station=picked)
      for killed_tr in st_to_remove:
         st.remove(killed_tr)
      art.remove()

      for gc in gc_lines:
         gc_line_label = gc[0].get_label()
         if gc_line_label == picked:
            gc[0].remove()

      fig.canvas.draw()
   
   fig.canvas.mpl_connect('pick_event', onpick)

   if showplot == True:
      plt.show()
   else:
      return m

#########################################################################
# Plot multiple traces ontop of eachother, aligned on a phase and windowed
#########################################################################
def compare(st, **kwargs):
  from seis_tools.seispy.data import phase_window

  plt.style.use('mystyle')
  window = kwargs.get('window',(-10,100))
  phase  = kwargs.get('phase',['P'])
  phase_list = kwargs.get('phase_list',['P'])
  taup_model = kwargs.get('taup_model','none')
  plot_all = kwargs.get('plot_all',False)
  phase_name = phase[0]

  if taup_model == 'none':
     taup_model = model

  if plot_all == True:
     arrs = taup_model.get_travel_times(source_depth_in_km = st[0].stats.sac['evdp'],
                                        distance_in_degree = st[0].stats.sac['gcarc'],
                                        phase_list=phase_list)

  i = 0
  for tr in st:
     if plot_all == False:
        windowed_trace = phase_window(tr,phases=phase,window_tuple=window,taup_model=taup_model)
        label_str = tr.stats.station+tr.stats.channel
        time = np.linspace(window[0],window[1],len(windowed_trace.data))
        plt.plot(time, windowed_trace.data,label=label_str)
        i += 1
     else:
        t_0 = 0.0
        t_e = tr.stats.npts * (1/tr.stats.sampling_rate)
        time = np.linspace(t_0,t_e,tr.stats.npts)
        plt.plot(time,tr.data)

  if plot_all == True:
     for arr in arrs:
        print arr
        plt.axvline(arr.time,label=arr.phase.name)

  plt.legend()
  plt.xlabel('time (s), windowed on '+phase_name) 
  plt.show()

def gmt_north_america(**kwargs):
   '''
   Give rf_data as ([values],[time])
   '''
   from gmtpy import GMT
   #get kwargs
   fname = kwargs.get('fname','USA_map.pdf')
   station_data = kwargs.get('station_data','none')
   quake_locs   = kwargs.get('quake_locs','none')
   rf_data      = kwargs.get('rf_data','none')
   header       = kwargs.get('header','none')

   #topo data
   etopo='/geo/home/romaguir/utils/ETOPO5.grd'
   topo_grad='/geo/home/romaguir/utils/topo_grad.grd'

   #colormap
   colombia='/geo/home/romaguir/utils/colors/colombia'

   region = '-128/-66/24/52'
   scale = 'l-100/35/33/45/1:30000000'

   #initialize gmt
   gmt = GMT(config={'BASEMAP_TYPE':'fancy',
                     'HEADER_FONT_SIZE':'14'})
                     #'COLOR_BACKGROUND':'-',  :setting this to '-' plots z are transparent
                     #'COLOR_FOREGROUND':'-'})

   #make colormap
   cptfile = gmt.tempfilename()
   gmt.makecpt(C=colombia,T='-4000/4950/100',Z=True,out_filename=cptfile,D='i')

   #make gradient
   #topo_grad = gmt.tempfilename()
   #gmt.grdgradient(etopo,V=True,A='100',N='e0.8',M=True,G=topo_grad,K=False)

   #plot topography
   gmt.grdimage( etopo,
                 R = region,
                 J = scale,
                 C = cptfile,
                 E = '200')
                 #I = '0.5')#topo_grad)

   #plot coastlines
   gmt.pscoast( R=region,
                J=scale,
                B='a10',
                D='l',
                A='500',
                W='thinnest,black,-',
                N='all')

   #plot stations
   if station_data != 'none':
      gmt.psxy( R=region,
                J=scale,
                B='a10',
                S='t0.1',
                G='red',
                in_rows = station_data )


   if quake_locs != 'none':
      eq_region = 'g'
      eq_scale  = 'E-100/40/2.25i'
      gmt.pscoast( R = eq_region,
                   J = eq_scale,
                   B = 'p',
                   A = '10000',
                   G = 'lightgrey',
                   W = 'thinnest',
                   Y = '3.5i',
                   X = '-0.5i' ) 
      gmt.psxy( R = eq_region,
                J = eq_scale,
                S = 'a0.05',
                G = 'blue',
                in_rows = quake_locs )

   #plot receiver function stack
   if rf_data != 'none':
      rf_region = '-0.20/0.20/0/100'
      rf_scale  = 'X1i/-5i'
      gmt.psxy( R = rf_region,
                J = rf_scale,
                #G = 'grey', #fill
                B = '0.25::/10:time (s):NE',
                W = 'thin',
                X = '9i',
                Y = '-3.5i',
                in_columns = rf_data )

      #plot shaded area for positive values
      vals = rf_data[0]
      time = rf_data[1]
      poly_vals_pos = []
      poly_time_pos = []
      for i in range(0,len(vals)):
         val_here = max(0,np.float(vals[i]))
         poly_time_pos.append(time[i])
         poly_vals_pos.append(val_here)

      poly_vals_pos.append(0.0)
      poly_time_pos.append(time[::-1][0])
      poly_vals_pos.append(0.0)
      poly_time_pos.append(time[0])
      gmt.psxy( R = rf_region,
                J = rf_scale,
                G = 'red',
                B = '0.25::/10:time (s):NE',
                W = 'thin',
                in_columns = (poly_vals_pos,poly_time_pos) )

      #plot shaded area for negative values
      '''
      vals = rf_data[0]
      time = rf_data[1]
      poly_vals_neg = []
      poly_time_neg = []
      for i in range(0,len(vals)):
         val_here = min(0,np.float(vals[i]))
         poly_time_neg.append(time[i])
         poly_vals_neg.append(val_here)

      poly_vals_neg.append(0.0)
      poly_time_neg.append(time[::-1][0])
      poly_vals_neg.append(0.0)
      poly_time_neg.append(time[0])
      gmt.psxy( R = rf_region,
                J = rf_scale,
                G = 'blue',
                B = '0.25::/10:time (s):NE',
                W = 'thin',
                in_columns = (poly_vals_neg,poly_time_neg) )
      '''

   #header_file = open('header_file','w')
   #header_file.write('<< EOF')
   #header_file.close()

   #write header information
   if header != 'none':
      stations_text = header['N_traces'] + ' traces in stack'
      events_text   = header['N_events'] + ' events'
      freq_text     = header['frequency']
      decon_text    = 'deconvolution method : ' + header['decon']

      gmt.pstext( in_rows = [(-96,60,12,0,1,'MC',stations_text)],
                  R = region,
                  J = scale,
                  N = True,
                  X = '-8.5i' )
      gmt.pstext( in_rows = [(-96,59,12,0,1,'MC',events_text)],
                  R = region,
                  J = scale,
                  N = True )
      gmt.pstext( in_rows = [(-96,58,12,0,1,'MC',freq_text)],
                  R = region,
                  J = scale,
                  N = True )
      gmt.pstext( in_rows = [(-96,57,12,0,1,'MC',decon_text)],
                  R = region,
                  J = scale,
                  N = True )

   #save figure
   gmt.save(fname)
