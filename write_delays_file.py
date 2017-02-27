import glob
from seis_tools.synth_tomo.delays import delays
from obspy.taup import TauPyModel

#name of output file-------------------------------------------------------------
output_name = 'r1a_delays_specfem.h5py'

#choose frequency bands----------------------------------------------------------
#freqmin and freqmax can be lists if you want to calculate delays for more than one band
freqmin = [1/25.0]
freqmax = [1/10.0]

#phases -------------------------------------------------------------------------
phases = ['P','S','SKS']

#model to compute 1d travel times------------------------------------------------
model = TauPyModel('prem')

#calculate delays and write output file------------------------------------------
for i in range(0,len(phases)):
    background_models = []
    plume_models = []
    plumepath='/r1a/r1a_d'

    if phases[i] == 'P' or phases[i] == 'S' or phases[i] == 'SS':
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d30/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d40/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d50/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d60/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d70/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d80/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d90/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'30/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'40/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'50/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'60/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'70/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'80/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'90/')
    elif phases[i] == 'SKS':
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d70/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d80/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d90/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d100/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d110/')
       background_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data/prem/prem_d120/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'70/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'80/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'90/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'100/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'110/')
       plume_models.append('/geo/work10/romaguir/seismology/synthetic_waveform_modelling/specfem/data'+plumepath+'120/')

    print '########################################################################################'
    print 'Working on phase ', phases[i]
    print '########################################################################################'

    #filter out event-receiver pairs that are out of the epicentral distance range of phase--------
    if phases[i] == 'P' or phases[i] == 'S':
        dist_range = (0,130)
    elif phases[i] == 'SS':
        dist_range = (25,140)
    elif phases[i] == 'SKS':
        dist_range = (65,140)

    #write delays to output file-------------------------------------------------------------------
    for j in range(0,len(freqmin)):   
        for k in range(0,len(plume_models)):
            print 'Working on event ',background_models[k]
            delta = background_models[k].split('prem_d')[1].split('/')[0]
            delays(background_models[k],plume_models[k],phases[i],freqmin[j],freqmax[j],
                   output_name,delta,distance_range=dist_range,plot=False,taup_model=model)
