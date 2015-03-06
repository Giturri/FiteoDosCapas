#import argparse
# -*- coding: utf-8 -*-

"""
Created on Wed Feb  5 10:01:05 2014

@author: Nicolas Abel Carbone / ncarbone@exa.unicen.edu.ar

Mathematical functions
"""

#imports
import math
import cmath
import scipy
import scipy.special as spsp
import numpy as np

#global variable declarations with default values
n_ref = 1.4
v = 299.792458000/n_ref
if n_ref > 1:
    A = 504.332889 - 2641.00214 * n_ref + 5923.699064 * n_ref**2 - 7376.355814 * n_ref**3 + 5507.53041 * n_ref**4 - 2463.357945 * n_ref**5 + 610.956547 * n_ref**6 - 64.8047 * n_ref**7
if n_ref <= 1:
    A = 3.084635 - 6.531194 * n_ref + 8.357854 * n_ref**2 - 5.082751 * n_ref**3 + 1.171382 *  n_ref**4

sep = 52.  #Slab thickness
ro = 0.  #Separation between the source and the optical axis    
data = []  #Array with experimental data
instru = []  #Array with response data 
first_nonzero_data = 0 #First nonzero value in the data array
last_nonzero_data = 4095 #Last nonzero value in the data array
#data_norm = []  #Array with normalized experimental data
data_count = [] #Array with experimental data counts
instru_count = [] #Array with response function data counts
temp_data = []  #Array with temporal x axis
max_temp = 50.
import cmath
import scipy
import scipy.special as spsp
def pre_calcs():
    """ Calculates A and v as a function of the refraction index, and the first and last non-zero values of the data
    
    TODO: more elegant way of doing this.    
    """
    global v
    global A
    global first_nonzero_data
    global last_nonzero_data    
    #baseline_sample = 50
    v = 0.0299792458000/n_ref

    n_out = 1.    #Air
    nn = n_ref/n_out

    A_factor=1.71148-5.27938*nn+5.10161*nn**2-0.66712*nn**3+0.11902*nn**4


    first_nonzero_data = np.nonzero(data_count)[0][0] #Calculate position of first nonzero value in data
    last_nonzero_data = np.nonzero(data_count)[0][-1] #Calculate position of last nonzero value in data
    #first_nonzero_instru = np.nonzero(instru)[0][0]
    #last_nonzero_instru = np.nonzero(instru)[0][-1]
    #baseline = sum(instru[first_nonzero_instru:(first_nonzero_instru+baseline_sample)])/baseline_sample
    #instru[first_nonzero_instru:last_nonzero_instru] = instru[first_nonzero_instru:last_nonzero_instru] - baseline


def smooth(x, window_len=10, window='hanning'): #Based on http://wiki.scipy.org/Cookbook/SignalSmooth
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

def funcion_teo_refl_2l (t, ups2, ua2, t0):

     global A
     global v

     #ariable declarations with default values
     i = complex(0,1)
     ua1 = .03         #Absorption coefficient of layer 1
     ua2 = ua2*10#.03         #Absorption coefficient of layer 2
     ups1 = 7.          #Reduced scattering coefficient of layer 1
     ups2 = ups2*10#10.          #Reduced scattering coefficient of layer 2
     lt1 = 1.          #Thickness of layer 1
     lt2 = 5.          #Thickness of layer 1
     n_ref1 = n_ref#1.4          #Refractive index of layer 1
     n_ref2 = n_ref#1.4          #Refractive index of layer 2
     radius = 20.          #cylinder radius
     rho = ro/10#4.              #Source-detector distance
     D1 = 1./(3.*ups1)           #Dispersion coefficient of layer 1
     D2 = 1./(3.*ups2)           #Dispersion coefficient of layer 2
     cn1 = v#0.0299792458000/n_ref1     #Light speed in layer 1
     cn2 = v#0.0299792458000/n_ref2     #Light speed in layer 2
     z0 = 1./ups1
     A_factor = 1.
     zb1 = 2.*A_factor*D1
     zb2 = 2.*A_factor*D2
     r_prime = radius+zb1
     #Arrays and other stuff
     n = 2
     ti = 0.
     Nt = 1024*n
     dt = 5
     tf = dt*Nt              #Final time in time array
     Nsn = 1500
     t_arr = np.arange(ti,tf,dt)
     t = t_arr
     f_arr = np.fft.fftfreq(Nt,d=dt)

     sn_arr = np.ndarray(shape=(Nsn))

     fl = np.ndarray(shape=(Nt))
     reff_2l = np.ndarray(shape=(Nt))


     """#Calculo A y r_prime
     n_out = 1.    #Air
     nn = n_ref1/n_out

     A_factor=1.71148-5.27938*nn+5.10161*nn**2-0.66712*nn**3+0.11902*nn**4"""

     A_factor = A
     zb1 = 2*A_factor*D1
     zb2 = 2*A_factor*D2

     r_prime = radius + zb1


     #Leo el archivo de ceros de Bessel

     j0roots = open("j0roots.dat",'r')

     for l in np.arange(0,Nsn,1):
     	sn_arr[l] = np.dot(float(j0roots.readline()),1./r_prime)

     j0roots.close()


     #Calculo la función de Green

     alfa1=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     alfa2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     exp1=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     exp2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     term1=np.ndarray(shape=(Nsn,Nt),dtype=complex)

     sh1=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     sh2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     exp3=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     den1=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     factor1=np.ndarray(shape=(Nsn,Nt),dtype=complex)

     num2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     cacho1=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     cacho2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     den2=np.ndarray(shape=(Nsn,Nt),dtype=complex)
     factor2=np.ndarray(shape=(Nsn,Nt),dtype=complex)

     green1=np.ndarray(shape=(Nsn,Nt),dtype=complex)

     k=0
     for bzero in sn_arr:

     	l=0
     	for f in f_arr:

     		omega = 2*math.pi*f

     		alfa1[k,l]=cmath.sqrt(ua1/D1+bzero**2+i*omega/(D1*cn1))
     		alfa2[k,l]=cmath.sqrt(ua2/D2+bzero**2+i*omega/(D2*cn2))

     		exp1[k,l] = cmath.e**(-alfa1[k,l]*abs(z0))
     		exp2[k,l] = cmath.e**(-alfa1[k,l]*(z0+2*zb1))
     		term1[k,l] = (exp1[k,l]-exp2[k,l])/(2*D1*alfa1[k,l])

     		if alfa1[k,l] == alfa2[k,l]:

     			green1[k,l] = term1[k,l]

     		else:

     			sh1[k,l] = cmath.sinh(alfa1[k,l]*(z0+zb1))
     			sh2[k,l] = cmath.sinh(alfa1[k,l]*(zb1))
     			exp3[k,l] = cmath.e**(alfa1[k,l]*(lt1+zb1))
     			den1[k,l] = D1*alfa1[k,l]*exp3[k,l]
     			factor1[k,l] = sh1[k,l]*sh2[k,l]/den1[k,l]

     			num2[k,l] = D1*alfa1[k,l]*n_ref1**2-D2*alfa2[k,l]*n_ref2**2
     			cacho1[k,l] = D1*alfa1[k,l]*cmath.cosh(alfa1[k,l]*(lt1+zb1))*n_ref1**2
     			cacho2[k,l] = D2*alfa2[k,l]*cmath.sinh(alfa1[k,l]*(lt1+zb1))*n_ref2**2
     			den2[k,l] = cacho1[k,l] + cacho2[k,l]
     			factor2[k,l] = num2[k,l]/den2[k,l]
	
     			green1[k,l] = term1[k,l] + factor1[k,l]*factor2[k,l]

     		l+=1
     	k+=1

     prefl1=np.ndarray(shape=(Nsn,Nt))
     prefl2=np.ndarray(shape=(Nsn,Nt))

     prefl1 = np.fft.ifft(green1,axis=1).real

     caca = 2.*(math.pi*r_prime)**2
     bessel_factor = np.ndarray(shape=(Nsn))
     bessel_factor = spsp.j0(sn_arr*rho)/spsp.j1(sn_arr*r_prime)**2
     for l in np.arange(0,Nt,1):
     	fl[l] = 0.

     prefl2 = bessel_factor*np.transpose(prefl1)
     fl = prefl2.sum(axis=1)/caca


     #Calculo la reflectancia

     reff_2l = fl/(2.*A_factor)

     return reff_2l


def funcion_fiteo_refl_klm(t,ups2,ua2,t0,back):

    array_teo = funcion_teo_refl_2l(temp_data, ups2, ua2, t0)
    array_teo = array_teo/array_teo.max() #Normalize the theoretical array
    array_conv = np.convolve(instru_count, array_teo, 'full') #Convolve with response function
    array_conv.resize(data_count.size) #Truncate resultiing array to the size of the experimental data
    array_conv = array_conv/array_conv.max() #Normalize the convoluted array
    array_conv[:first_nonzero_data] = 0 #Set to zero the same positions as in the data array
    array_conv[last_nonzero_data:] = 0 #Set to zero the same positions as in the data array
    return np.interp(t, temp_data, array_conv) + back #Return the final value for t, plus a background constant level
