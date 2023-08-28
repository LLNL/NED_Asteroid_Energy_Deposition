# ----------------------------------------------------------------------
# Copyright (c) 2023, Lawrence Livermore National Security, LLC
#
# LLNL-CODE-853602
#
# SPDX-License-Identifier: (BSD-3-Clause) 
#
# This file integrates the total deposition in a 2D spherical problem.
# ----------------------------------------------------------------------

from energyDep import *
import scipy.integrate

def RadLowlim(theta):
    """ TRad in m, totdepth is in cm"""
    return TRad - totdepth/100.0

def RadUplim(theta):
    """ TRad in m"""
    return TRad

def EdepfuncIntegrand(radius, theta, Material, BB, Yield, Tsrc, Porosity, TRad, HOB):
    """
    radius in m
    theta in radians
    Material in ["SiO2", "Forsterite","Iron", "Ice"]
    BB in [1.0, 2.0]
    Yield in kt
    Tsrc in ns
    Porosity - dimensionless, in range (0, 1.0)
    TRad in m
    HOB in m
    returns energy in Perg/m, assuming integral over r in meters
    """
    d      = TRad - radius  # in m
    length = math.sqrt((TRad+HOB)**2 + TRad**2 - 2.*(TRad+HOB)*TRad*math.cos(theta)) # in m
    beta   = math.cos(math.asin(math.sin(theta)*(HOB+TRad)/length))
    Flx    = Yield*beta/(4*math.pi*(length**2)) # kt/m^2
    dE     = Edepfunc(d*100.0, Material, BB, beta, Flx, Tsrc, Porosity) # Perg/cm^3
    return dE*1e6*2.*math.pi*radius**2 * math.sin(theta) # dE*1e6 gives Perg/m^3 to match integral in m

Materials = ['SiO2','Forsterite','Ice','Iron']
BB    = 2.0
Tsrc  = 50.
TRad  = 400.0 # in m
HOB   = 10.0  # in m
Yield = 1000.0 # in kt
Porosity = 0.21 # 0.32
AngRes = 200
Depthres = 5.e-5 # in cm
totdepth = 10.0 # in cm 
theta_max = math.acos(TRad/(TRad + HOB))# math.acos(5500./(5501.)) #math.acos(TRad/(TRad + HOB))
alpha_max = math.asin(TRad/(TRad + HOB))
Esum=0.0
print("Energy intercepted by asteroid ", Yield*0.5*(1. - math.cos(alpha_max)))

# Find depth of deposition on axis
# This helps the integration routine find where the integrand is non-zero
beta = 1.0
Flx  = Yield/(4*math.pi*(HOB**2)) # kt/m^2
dE   = 1.0
d    = Depthres
while dE > 1.e-80:
    dlast = d
    d = 1.5*d
    dE=Edepfunc(d, Materials[0], BB, beta, Flx, Tsrc, Porosity)
totdepth = d

print("Starting integral, this may be slow, be patient.")
EsumInt = scipy.integrate.dblquad(EdepfuncIntegrand, 0., theta_max, RadLowlim, RadUplim, args=(Materials[0], BB, Yield, Tsrc, Porosity, TRad, HOB))
print("Integrated deposited energy:    ", EsumInt[0]/4.184e4, "kt  +- ", EsumInt[1]/4.184e4, " kt")

for t in range(1,AngRes,2):
    #if t % 5 ==0:
    #    print(t)
    depflag=1
    theta=theta_max*t/AngRes
    theta0=theta_max*(t-1)/AngRes
    theta1=theta_max*(t+1)/AngRes
    length=math.sqrt((TRad+HOB)**2+TRad**2-2.*(TRad+HOB)*TRad*math.cos(theta))
    length0=math.sqrt((TRad+HOB)**2+TRad**2-2.*(TRad+HOB)*TRad*math.cos(theta0))
    length1=math.sqrt((TRad+HOB)**2+TRad**2-2.*(TRad+HOB)*TRad*math.cos(theta1))
    beta=math.cos(math.asin(math.sin(theta)*(HOB+TRad)/length))
    textra=abs(length1-length0)/2.99792458e2
    Flx=Yield*beta/(4*math.pi*(length**2))
    for d in np.arange(Depthres,totdepth,2.0*Depthres):
        
        dE=Edepfunc(d, Materials[0], BB, beta, Flx, Tsrc+textra, Porosity)
        if dE<1.e-10:
            depflag=0
            break
        vol=2.*math.pi*(((TRad*100.-(d-Depthres))**3)-((TRad*100.-(d+Depthres))**3))*(math.cos(theta0)-math.cos(theta1))/3.0
        Esum=Esum+dE*vol
    if depflag==1:
        print("totdepth (%e) not deep enough, quitting" % totdepth)
        break

print("Zone-centered deposited energy: ", Esum/4.184e4,"kt") 

