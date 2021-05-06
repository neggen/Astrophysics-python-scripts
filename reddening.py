import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.io import fits
import math
import sys


'''
This script corrects for galactic reddening. To run, find the reddening 
coefficient (Av) along line of sight using online database, and use latest calculated Rv for milkyway (3.1 as of 2020)
''' 

def optical_ext(wavelength,Av,Rv):
	'''
	calculates optical extinction using empirical relations calculated from literature. google calzetti galactic extinction for reference. 
	'''
	x=(wavelength/10000.)**(-1.)
	print(x[0],x[-1])
	y=x-1.82
	a=1. + 0.17699*y - 0.50447*y**2. - 0.02427*y**3. + 0.72085*y**4. + 0.01979*y**5. - 0.7753*y**6. + 0.32999*y**7.
	b=1.41338*y+2.28305*y**2.+1.07233*y**3.-5.38434*y**4.-0.62251*y**5.+5.3026*y**6.-2.09002*y**7.
	print(a[0],b[0])
	print(a[-1],b[-1])
	return (a+b/Rv)*Av

#Update coefficients
Av = 0.1228
Rv = 3.1

#load in datacube
cube=fits.open("CombinedCube.fits")
data=cube[1].data

#redshift to object
z=0.0040781

#calculate wavelength along spectral axis
nx=cube[1].header['NAXIS3']
dx=cube[1].header['CD3_3']
x0=cube[1].header['CRVAL3']
print(x0,dx,nx)
wavelength=np.array([dx*(l)+x0 for l in range(0,nx)])

#calculate and plot reddening 
red=optical_ext(wavelength,Av,Rv)
plt.figure
plt.plot(wavelength,10**(red/2.5))
# plt.plot(wavelength,red)
plt.show()

#sanity check
print(data[0,0,0])

#correct reddening and save to file
cube[1].data=data*10**(red[:,None,None]/2.5)
cube.writeto("CombinedCube_corr.fits",overwrite=True)
