from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import math
import pylab as P
from scipy import ndimage
import sys

'''
This Script takes in a list of datacubes in command line, aligns them in
the spectral dimension, and then combines them together by using the median. 
Median was chosen to suppress the influence of outliers that exist in the 
datacubes, and the variance is propagated correctly as well.
'''

if len(sys.argv) < 3:
	print("Incorrect number of arguments; please provide file names")
	sys.exit(1)

#Different exposures have different spectral coverage. Choose length along spectral axis that is the union of all exposures coverage.
l2=5940

#Load in first data-cube
cube1 = fits.open(sys.argv[1])
data1 = cube1[1].data
dq = cube1[3].data
var1 = cube1[2].data
x0 = cube1[1].header['CRVAL3']
dx = cube1[1].header['CD3_3']
nx1 = cube1[1].header['NAXIS3']
x1 = np.array([dx*(x)+x0 for x in range(0,l2)])

#Insert zeros for spectral range first datacube doesn't cover
temp = np.zeros((l2-len(data1),49,33))
tempdq = temp[:]+1
data1 = np.append(data1,temp,axis=0)
var1 = np.append(var1,temp,axis=0)
dq = np.append(dq,tempdq,axis=0)
dq1 = np.copy(dq)

##masking out chip gaps (138)
dq1[((4383. < x1) & (x1 < 4450.)),:,:] = 1
mask1 = np.ma.make_mask(dq1)
#mask1=~mask1
mdata1 = np.ma.masked_array(data1,mask=mask1)
mvar1 = np.ma.masked_array(var1,mask=mask1)

#initalize set of datacubes that will be loaded in below
scistack = np.ma.masked_array(np.zeros((len(sys.argv)-1,l2,49,33)),np.zeros((len(sys.argv)-1,l2,49,33)))
varstack = np.ma.masked_array(np.zeros((len(sys.argv)-1,l2,49,33)),np.zeros((len(sys.argv)-1,l2,49,33)))

#load first datacube into stack
scistack[0] = mdata1
varstack[0] = mvar1

#load in rest of datacubes
for i in range(0,(len(sys.argv) - 2)):
	print("File: "+sys.argv[i+2])
	cube = fits.open(sys.argv[i+2])
	data = cube[1].data
	dq = cube[3].data
	var = cube[2].data
	x0 = cube[1].header['CRVAL3']
	dx = cube[1].header['CD3_3']
	nx = cube[1].header['NAXIS3']
	print(len(data),len(data1),nx,nx1)
	x2 = np.array([dx*(x)+x0 for x in range(0,l2)])
	print(x1[0],x2[0])
	temp = np.zeros((l2-len(data),49,33))
	tempdq = temp[:]+1
	data = np.append(data,temp,axis=0)
	var = np.append(var,temp,axis=0)
	dq = np.append(dq,tempdq,axis=0)
	##Mask out the chip gaps: order of inputs must be in numerical order
	if i > 0:
		dq[((4475 < x2) & (x2 < 4530)),:,:] = 1
	if i == 0:
		dq[((4383 < x2) & (x2 < 4450)),:,:] = 1
	

	#resample datacube along spectral axis to line up for combining later.
	#Data has already been aligned spatially with @align.py	
	fsci = interp1d(x2,data,axis=0,kind='linear',fill_value='extrapolate')
	fvar = interp1d(x2,var,axis=0,kind='linear',fill_value='extrapolate')
	fdq = interp1d(x2,dq,axis=0,kind='nearest',fill_value='extrapolate')

	intersci = fsci(x1)
	intervar = fvar(x1)
	interdq = fdq(x1)
	intermask = np.ma.make_mask(interdq)
	#intermask=~intermask
	
	#Save resampled datacube to file if desired to double check validity
	# cube[1].data=intersci
	# cube[2].data=intervar
	# cube[3].data=interdq
	# cube.writeto("testinginterp"+str(i)+".fits",overwrite=True)

	#save resampled datacube into the stack of datacubes
	scistack[i+1] = np.ma.MaskedArray(intersci,intermask)
	#print(mdata2[2741])
	#print(scistack[i+1,2741])
	varstack[i+1] = np.ma.MaskedArray(intervar,intermask)

#Sanity checks
print(np.shape(scistack)[0])
print(scistack[-1,2741,20,15],intersci[2741,20,15])
ah = scistack[-3:]
print(ah.shape)
aah = np.ma.median(ah,axis=0)
bah = np.ma.masked_array(np.zeros((3,l2,49,33)),np.zeros((3,l2,49,33)))
for i in range(2):
	bah[i] = scistack[i]
bah[2] = aah
print(bah.shape)

#combine datacubes into one. Masked arrays automatically ignore masked pixels from each array in the pixel by pixel median. 
average_sci = np.ma.median(scistack,axis=0)
averagesci = np.ma.filled(average_sci, fill_value=np.nan)
average_var = np.ma.median(varstack,axis=0) / (np.shape(varstack)[0])
averagevar = np.ma.filled(average_var, fill_value=np.nan)
cube1[1].data = averagesci
cube1[2].data = averagevar
cube1[3].data = dq
cube1.writeto("CombinedCube.fits",overwrite=True)
