# ----------------------------------------------------------------------
# Copyright (c) 2023, Lawrence Livermore National Security, LLC
#
# LLNL-CODE-853602
#
# SPDX-License-Identifier: (BSD-3-Clause) 
#
# energyDepFile
# ----------------------------------------------------------------------

import math
import numpy as np
import os
import sys

from scipy.stats import semicircular   # semi-circular PDF, pdf(x) = (2/pi)*sqrt(1 - x^2)
from scipy.special import expit        # logistic sigmoid function, 1/(1+exp(-x))

def anglefunc(M, *p):
    """
    How to scale fluence for angle of incidence.
    inputs:
      M is an array of length 3
        cos(angle of incidence)
        log10(fluence), fluence in kt/m^2
        tsrc - length of source in sh = 10 ns
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M #x = angle, y = fluence, z = tsrc

    f = p[0] + p[1]*x + p[2]*x**2 + y*p[3] + p[4]*(y**2) + p[5]*(y**2)*(x**2) + p[6]*(y**3) + p[7]*(y**3)*(x**2) + p[8]*z + p[9]*z**2 + p[10]*(x**2)*(z**2) + p[11]*z*x**2
    return f

def xfunc(M,*p):
    """
    The horizontal radius of the ellipse.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = (10**(p[0] + p[1]*x + y*p[2] + p[3]*(y**2) + p[4]*(y**2)*x + p[5]*(y**3) + p[6]*(y**3)*(x**2)))*(1.0 + p[7]*np.exp((p[8]*x*x+p[9]*x)*(z-p[10]))*np.exp(-(p[11]*x*x)*(y-p[12])**2))
    return f
    
def yfunc(M,*p):
    """
    Scaling factor used to determine the height of the ellipse.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = 10**(p[0] + y*p[1] + p[2]*y*x**3 + p[3]*(y**2) + p[4]*(y**2)*(x**3) + p[5]*(y**3))
    return f
    
def exfunc(M,*p):
    """
    Linked to the decay rate of the tail, 
    and the most temperamental fitting parameter of the profile. 
    It was configured as a fraction to prevent the fitting routine 
    from converging on unrealistic values.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = 10**(p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 +y*p[4] + p[5]*(y**2) + p[6]*(y**3)*x + p[7]*(y**3)*(x**2) + p[8]*(y**3)*(x**3))
    return f

def sqfunc(M,*p):
    """
    Height of the "square" beneath the half-ellipse. 
    Must equal the 1/x function at the horizontal ellipse diameter for continuity.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = p[0] + p[1]*x**2 + y*p[2] + p[3]*y*x**2 + p[4]*(y**2) + p[5]*(y**2)*(x**2) + p[6]*(y**3)
    return abs(f)

def bfunc(M, *p):
    """
    Represents the constant height of the shockwave: 
    taken from the $\mathcal{M}x +\mathcal{B}$ linear function 
    before $\mathcal{M}$ was deemed unnecessary for the profile fits.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = (10**(p[0] + p[1]*x + p[2]*x**2 + y*p[3] + p[4]*y*x + p[5]*(y**3))) * (1.0 + p[6]*z + p[7]*(y**2)*(z**3) + p[8]*(y**3)*(z**3) + p[9]*x*z )
    return f
    
def cutfunc(M, *p):
    """
    The depth where the shockwave energy density sharply drops; 
    selected prior to fitting and used as fixed parameter 
    to stabilize the fitting results.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M

    f = (10**(p[0] + p[1]*x + y*p[2] + p[3]*y*x + p[4]*y*x**2 + p[5]*(y**2)))*(1+p[6]*z + p[7]*z**3 + p[8]*(y**2)*z + p[9]*x*z)
    return f
    
def exptfunc(M):
    """
    Defines how quickly the shockwave drops to 0 using the expit function in Python, 
    1/(1 + e^{-x}). It is set to 1e4 in all cases except Ice-2keV, where the parameter is allowed to float.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns scale factor
    """
    x,y,z = M
 
    f = (10**(-0.1017 + y*(-0.3407) + 0.03137*y*x + (-0.003059)*y*x**2 + 0.0001539*y*x**3 + 0.05632*(y**2) + 0.009446*(y**3)))*(1.0 -0.0988*z + 0.7562*(y**2)*z + 0.1621*(y**3)*z)
    return f

def scalefunc(M, *p):
    """
    Defines overall scaling of the deposition.
    inputs:
      M is an array of length 3
        tsrc - length of source in sh = 10 ns
        log10(scaled fluence), fluence in kt/m^2
        porosity
      p is a list of coefficients for the given material and BB temperature

    returns depostion in jerks/cm^3
    """
    x,y,z = M   

    f = (p[0] + p[1]*x**3 +y*p[2] + p[3]*y*x + p[4]*y*x**2 + p[5]*(y**2) + p[6]*(y**2)*x + p[7]*(y**2)*(x**2) + p[8]*(y**3) + p[9]*(y**3)*x + p[10]*(y**3)*(x**2) + p[11]*(y**4))*(1.0 + p[12]*z + p[13]*(y**2)*z +p[14]*x*z)*(1.0 + p[15]*z**3 + p[16]*y*z + p[17]*(y**2)*(z**2) + p[18]*(y**2)*(z**3) + p[19]*x*z**2)
    return abs(f)

def test_dist(x, scale, xscale, yscale, exfrac, square, b, cutoff, expt):
    """
    Combine all the terms together for the full depostion.
    Inputs:
      x is depth in cm
      scale is result from scalefunc
      xscale is result from  xfunc
      yscale is result from yfunc
      exfrac is result from exfunc
      square is result from sqfunc
      b      is result from bfunc
      cutoff is result from cutfunc
      expt   is result from exptfunc or 1e4
    """
    rv = semicircular()
    exscale = (1.0 - exfrac)*2.0*xscale
    eyscale = (square - b)*(2.0*xscale - exscale)
    if square - b < 0.0:
        eyscale = 0.0
        b = square
    y = np.piecewise(x, [x < 2.0*xscale, x >= 2.0*xscale ], [lambda x: (math.pi/2)*rv.pdf((x)/xscale - 1)*yscale + (eyscale/((2*xscale) - exscale)+b), lambda x: (eyscale/(x - exscale)+b)*expit((cutoff - x)*abs(expt))])
    return y*scale

def Edepfunc(z, Mat, BB, cosang, Flx, Tsrc, Porosity):
    """
    internally the functions work in units based on gram-cm-shake, where a shake = 1e-8 seconds
     1 jerk = 1e16 erg, = 10 Perg
     1 kt   = 4184 jerks
     Flx leave in kt/m^2 but can use *= 0.4184 to convert kt/m^2 to jerks/cm^2

     inputs:
       z depth from surface in cm
       Mat in ['SiO2','Forsterite','Ice','Iron']
       BB  in [1.0, 2.0] - source temperature
       cosang - cosine of angle of incidence
       Flx - fluence in kt/m^2
       Tsrc - length of source in ns
       Porosity - potosity in range (0.0, 1.0)
    output
      energy density - Perg/cm^3
    """
    Tsrc *= 0.1   # convert ns to sh

    if Mat == 'SiO2' and BB == 1.0:
        
        xpar  = [-0.821, -0.01062,  0.7965,  0.0663, 0.002539, 0.009045, 2.236e-05, 0.8703, 3.32e-08, 5.603, 0.7576, 0.4493, 0.5161]
        ypar  = [-1.241,  0.4446, 0.0001668,  0.1786, 4.368e-05, 0.03968]
        expar = [-2.7, -0.06186, 0.01893, -0.001351,  -1.028, -0.09487, 0.0008123, 4.22e-05, -1.01e-05]
        sqpar = [0.06365, -7.282e-05, 0.02514, -7.018e-05, 0.004575, -1.448e-05, 0.0004084]
        bpar  = [-0.8981, -0.2591, 0.01348,  0.9919, 0.001909, -0.01093,  0.5743, 4.216, 0.6501, 0.1582]
        cutpar = [-0.3001, 0.03123,  0.4528, -0.02299, 0.001573, 0.02443, -0.6129, -0.1698, -0.005703, -0.0005767]
        scalepar = [1.437, -0.0002888, -0.4738,  0.2294, -0.01776, -0.2126, 0.07529, -0.006181, 0.01544, 0.006517, -0.0006078, 0.007554,   0.1144, -0.004564, -0.008957, -0.4666, -0.2109, -0.1392,   0.195, 0.07671]
        anglepar = [0.3936,   1.097, -0.4882, 0.008077, -0.07355, 0.07821, -0.01495, 0.01566, -0.02615, 0.001539, -0.001541, 0.02617]

    if Mat == 'SiO2' and BB == 2.0:
        
        xpar  = [-0.5752, -0.01313,  0.7952, 0.02273, 0.004744, 0.007201, 5.285e-05, 0.7297, 6.54e-08, 4.715, 0.8083, 0.3627, -0.15]
        ypar  = [-1.071,  0.4621, 0.0001761,  0.0998, 6.014e-05,  0.0274]
        expar = [-0.8663, -0.06759, 0.01458, -0.0009612,  -0.315, -0.009217, -0.0004816, 0.0001639, -1.341e-05]
        sqpar = [0.05872, -0.0001176, 0.00688, -7.248e-05, -0.003936, -1.172e-05, -0.0006042]
        bpar  = [-0.8316, -0.2511, 0.01188,  0.9452, -0.004228, -0.02063, -2.31, 2.214, 0.4393, 0.56]
        cutpar = [-0.1992, 0.03393,   0.434, -0.02525, 0.001791, 0.01596, -0.646,  -0.219, -0.005971, 0.006443]
        scalepar = [1.62, -0.000382,  0.4064,  0.2057, -0.02149,  0.4235, 0.06424, -0.00848,  0.1375, 0.007716, -0.001106, 0.01572, 0.163, -0.005791, -0.01273,-0.5947, -0.5759, -0.4853,  0.5013, 0.03824]
        anglepar = [0.3016,   1.231, -0.5288, 0.01353, -0.0463, 0.05415, -0.0132, 0.01442, -0.03364, 0.001878, -0.001882,  0.0337]
        
    if Mat == 'Forsterite' and BB == 1.0:
        
        xpar  = [-0.8828, -0.01102,  0.8733,  0.1188, 0.00292, 0.01931, 3.461e-05, 0.7723, 2.55, 2.9, 0.7555, 0.2947, 0.8421]
        ypar  = [-1.045,  0.4522, 0.0001805,  0.1289, 4.447e-05, 0.02758]
        expar = [-2.492, 0.06824, -0.01225, 0.0007029,  -0.916, -0.08725, 0.001397, -0.000222, 1.222e-05]
        sqpar = [0.08657, -9.268e-05, 0.04068, -8.083e-05, 0.009029, -1.558e-05, 0.0008339]
        bpar  = [-0.8612, -0.2633, 0.01218,   1.097, -0.003769, -0.01605,  0.5394, 7.51, 1.419, 0.2365]
        cutpar = [-0.2937, 0.03087,   0.476, -0.0239, 0.001566, 0.03159, -0.6729, -0.06639, -0.009951, -0.004046]
        scalepar = [1.374, -0.0002621, -0.9945,  0.3332, -0.02362, -0.6901,  0.1484, -0.01059, -0.1148, 0.01842, -0.001363, -0.003327, 0.08486, -0.003644, -0.006243, -0.3497, -0.1602, -0.05312,  0.1125, 0.07712]
        anglepar = [0.3516,   1.131, -0.4795, 0.008258, -0.0773, 0.08212, -0.01712, 0.01786, -0.01648, 0.0006668, -0.0006605,  0.0164]

    if Mat == 'Forsterite' and BB == 2.0:
        
        xpar  = [-0.6415, -0.008679,  0.8131, 0.03811, 0.004115, 0.01179, 5.363e-05, 0.6665, 1.78e-07, 4.968, 0.7986, 0.3732, -0.1661]
        ypar  = [-1.004,  0.3339, 9.018e-05, 0.03465, 2.78e-05, 0.01906]
        expar = [-1.101, -0.07457, 0.01106, -0.0005612, -0.6614, -0.09991, -0.003416, 0.0006064, -3.306e-05]
        sqpar = [0.1033, -0.00013, 0.05112, -7.151e-05, 0.01219, -1.044e-05, 0.001317]
        bpar  =  [-0.7259, -0.2514, 0.01074,   1.139, -0.01383, -0.03216, 1.901, 6.808, 1.739, 0.09991]
        cutpar =  [-0.205, 0.03632,  0.4399, -0.02482, 0.00175,  0.0198, -0.6949, -0.1273, -0.009989, 0.002671]
        scalepar = [1.515, -0.0002794, -0.0701,  0.2089, -0.01775, -0.04096, 0.06932, -0.006942, -0.01342, 0.008389, -0.0009254, -0.001657, 0.1385, -0.006208, -0.009919,-0.5577, -0.3214, -0.3319,  0.4605, 0.06743]
        anglepar = [0.3193,   1.255, -0.5703,  0.0131, -0.04415, 0.05165, -0.01188, 0.01303, -0.04153, 0.002697, -0.002698, 0.04156]

    if Mat == 'Ice' and BB == 1.0:
        
        xpar  = [0.1313, -0.01121,  0.6872, -0.1093, 0.003511, -0.01501, 5.844e-05, 0.7005, 1.494e-06, 4.439, 0.8005,   0.02265,  -0.5356]
        ypar  = [-1.588,  0.3776, 2.838e-05,  0.2719, -4.383e-05,  0.1353]
        expar = [-1.333, 0.05571, -0.01946, 0.001206,  -0.281, 0.04659, 0.005701, -0.001255, 7.223e-05]
        sqpar = [0.03391, -3.008e-05, 0.01676, -1.136e-05, 0.004275, -2.854e-07, 0.0005034]
        bpar  = [ -1.038, -0.1488, 0.007594,  0.8216, 0.00976, -0.01044, 0.8312, 3.156, 0.7005, 0.09776]
        cutpar = [0.2192,  0.0296,  0.5659, -0.03467, 0.002565, 0.02833, -0.6047, -0.3111, -0.004676, 0.008665]
        scalepar = [1.136, 0.0001099, -0.4869,  0.0333, -0.001395, -0.2043, -0.01053, 0.0008495, -0.02507, -0.001785, 0.0001751, 0.001147, 0.1615, -0.007666, -0.01186,-0.9927, -0.1862, -0.1368,  0.2472, 0.04534]
        anglepar = [0.4563,     1.1, -0.5611, -0.01532, -0.02545, 0.01666, -0.002016, 0.0006665, -0.02675, 0.001373, -0.001375, 0.02679]

    if Mat == 'Ice' and BB == 2.0:
        
        xpar  = [0.3258, -0.00887,  0.7231, -0.1675, 0.003119, -0.02308, 2.959e-05, 0.9745, 1.058e-06,   3.455,  0.8318,  0.05, -2.197]
        ypar  = [ -1.568,  0.3209, 0.0001757, 0.08257, 5.144e-05, 0.09825]
        expar = [-1.085, -0.1233, 0.01717, -0.0008327,  -1.153, -0.2232, -0.006302, 0.0009429, -4.656e-05]
        sqpar = [0.03562, -5.418e-05, 0.01245, -2.887e-05, 0.0005093, -3.858e-06, -0.0001222]
        bpar  = [ -2.082, -0.1028, 0.004681,  0.3755, 0.009084, -0.008745, 1.08, 5.33, 1.309, -0.2584]
        cutpar = [0.3666, 0.02416,  0.6587, -0.03467, 0.002046, 0.04117, -0.6547, -0.1758, -0.01227, 0.002046]
        scalepar = [1.052, 0.0002935, -0.9585, -0.01884, 0.003326, -0.8081, -0.08345, 0.004746,  -0.394, -0.0151, 0.0007523, -0.05733, 0.05345, -0.004118, -0.002399, -1.087, -0.3361, -0.1232,  0.2408, 0.08083]
        anglepar = [0.4768,    1.02, -0.5014, -0.01265, -0.04004, 0.03275, -0.008026, 0.006904, -0.01532, 0.0006698, -0.0006733, 0.01537]

    if Mat == 'Iron' and BB == 1.0:
        
        xpar  = [-1.25, -0.009912,  0.8485,  0.1216, 0.00208, 0.01725, 1.892e-05, 0.333, 3.96e-06, 6.148, 0.6787, 0.5849, 0.6671]
        ypar  = [-0.8209, -0.04753, 0.0002007,  -0.202, 5.326e-05, -0.02643]
        expar = [-2.258,  -0.065, 0.008586, -0.0003636,    -0.6, -0.0272, -0.0002216, -9.747e-06, 2.48e-06]
        sqpar = [0.2343, -0.0001397,  0.1507, -8.809e-05,  0.0408, -1.43e-05, 0.004077]
        bpar  = [-0.8231,   -0.29, 0.01378,   1.223, -0.005976, -0.02038, 1.638, 10.97, 2.029, 0.2081]
        cutpar = [-0.5032,  0.0326,   0.495, -0.02465, 0.001607, 0.03963, -0.7158,  0.0544, -0.01419, -0.009288]
        scalepar = [1.309, -0.000246, -0.9526,  0.3657, -0.02455, -0.7965,  0.1777, -0.01146, -0.1622, 0.02277, -0.001449, -0.009181, 0.04207, -0.006174, -0.00293, -0.3181, -0.2691, -0.1021,  0.1876, 0.08552]
        anglepar = [0.3421,   1.005, -0.3487, -0.004707, -0.08001, 0.07733, -0.01611,  0.0157, -0.01626, 0.0008547, -0.0008516, 0.01619]

    if Mat == 'Iron' and BB == 2.0:
    
        xpar  = [-1.253, -0.007975,  0.8386,  0.1472, 0.002349, 0.02395, 2.386e-05, 0.401, 1.12e-05, 4.989, 0.8198, 0.1354, 0.6349]
        ypar  = [-0.791, -0.01914, 0.0002803, -0.1709, 7.717e-05, -0.01573]
        expar = [-2.146,  -0.233, 0.04379, -0.002482, -0.7942, -0.07962, -0.004448, 0.0008916, -5.169e-05]
        sqpar = [0.2256, -0.0002188,  0.1341, -0.0001667, 0.03625, -2.9e-05, 0.003856]
        bpar  = [-0.8571, -0.2754, 0.01294,    1.16, -0.00567, -0.01937, 1.644, 11.19, 2.195, 0.1806]
        cutpar = [-0.5152, 0.03464,  0.4557, -0.02401, 0.001625, 0.03343, -0.6894, -0.009196, -0.01232, -0.007376]
        scalepar = [1.287, -0.0002182, -0.9068,  0.3268, -0.02227, -0.7455,  0.1582, -0.01046, -0.1419, 0.02048, -0.001346, -0.006235, 0.03121, -0.002817, -0.003192, -0.3163,  -0.209, -0.07131,  0.1555,  0.0939]
        anglepar = [0.3034,   1.073, -0.3737, 0.00931, -0.07177, 0.07716, -0.01607, 0.01689, -0.02651, 0.001541, -0.001542, 0.02649]
    
    # set up arguments for anglefunc
    xa = np.zeros(3)
    xa[0] = cosang
    xa[1] = math.log10(Flx)
    xa[2] = Tsrc

    xp = np.zeros(3)
    xp[0] = Tsrc
    EdgeScale = 1.0
    xp[1] = Flx*anglefunc(xa, *anglepar)
    if xp[1] <= 0:
    	return 0.0
    elif xp[1] < 1.0e-4:
        EdgeScale = xp[1]*1.0e4  
        xp[1] = -4.0  
    else:
        xp[1] = math.log10(xp[1])
    
    xp[2] = Porosity
    par = [scalefunc(xp, *scalepar), xfunc(xp,*xpar), yfunc(xp,*ypar), exfunc(xp,*expar), sqfunc(xp,*sqpar),  bfunc(xp,*bpar), cutfunc(xp,*cutpar)]
    # Add final argument for how quickly to drop to zero, only Ice-2 KeV changes this.
    if (Mat == 'ice') and (BB == '2keV'):
        par.append(exptfunc(xp))
    else:
        par.append(1.0e4)
    
    return 10.0*test_dist(z*(1-Porosity), *par)*(1-Porosity)*EdgeScale # convert jerk/cm^3 to Perg/cm^3

