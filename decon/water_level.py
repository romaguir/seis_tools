import numpy as np
import matplotlib.pyplot as plt

def water_level(a,b,alpha=0.1,plot=False):
    time = np.linspace(0,100,len(a))

    #Convert to frequency domain-------------------------------
    a_omega = np.fft.fft(a)
    b_omega = np.fft.fft(b)

    #Perform division------------------------------------------
    F_omega = b_omega*b_omega.conjugate()
    Phi_ss  = np.maximum(F_omega, alpha*( np.amax(F_omega)))
    H_omega = ((a_omega*b_omega.conjugate()) / Phi_ss)

    #Convert back to time domain-------------------------------
    #rf = np.zeros(len(H_omega))
    rf = np.fft.ifft(H_omega)
    return np.real(rf)

    #Plots-----------------------------------------------------
    if plot==True:
       plt.figure(figsize=(7,4))
       plt.plot(a)
       plt.plot(b)
       plt.plot(time,rf)
       plt.show()
