from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import math
import pylab as P
from scipy import ndimage
from stsci import imagestats
from mpl_toolkits.mplot3d import Axes3D
import sys

'''
Takes an input of a list of 2 datacubes in command line and aligns them 
spatially by correlating the images at the various nebular lines, and finding the offset
for the pixel with the largest correlation. Use nebular lines as they have most
structure and are brightest. This script does not calculate any rotation when aligning 
as the data all had the same orientation on the sky. 
'''

def loadfits(filename):
	cube = fits.open(filename)
	data = cube[1].data
	var = cube[2].data
	dq = cube[3].data
	x0 = cube[1].header['CRVAL3']
	dx = cube1[1].header['CD3_3']
	nx = cube1[1].header['NAXIS3']
	return cube,data,var,dq,x0,dx,nx

def find_emission_line_inds(linelist,x0,dx):
	lines = np.rint((linelist-x0)/dx).astype(int)
	return lines



#Check for multiple file inputs
if len(sys.argv) < 3:
	print("Incorrect number of arguments; please provide file names")
	sys.exit(1)

#Read in first datacube
cube1,data1,var1,dq1,x0,dx,nx = loadfits(sys.argv[1])
x1=[dx*(x)+x0 for x in range(0,nx)]

#find indicies for the emission lines along spectral axis
line_list = np.array([3884.3765,3904.5861,3984.715,4118.5283,4358.2135,4381.1539,4730.6792,4759.6404,4978.9637,5026.5423,5899.7814])
lines1 = find_emission_line_inds(line_list,x0,dx)


#Read in second datacube
cube2,data2,var2,dq2,x0,dx,nx = fits.open(sys.argv[2])
x2=[dx*(x)+x0 for x in range(0,nx)]

#find indicies again
lines2 = find_emission_line_inds(line_list,x0,dx)

#sanity check
print(lines1,lines2)

positions=[]
shift=[]
for k in np.ndenumerate(lines2):
	#get both indicies for the two cubes being aligned
	j=int(lines1[k[0]])
	i=int(k[-1])
	
	#more sanity checks
	print(i)
	print(j)

	#sum over the spectral range of each line to first get image of the entire nebular line
	#Then find the mean pixel value of each resulting image
	mean1 = imagestats.ImageStats(np.sum(data1[j-5:j+5,:,:],axis=0),nclip=0).mean
	mean2 = imagestats.ImageStats(np.sum(data2[i-5:i+5,:,:],axis=0),nclip=0).mean
	#subtract the mean from the image
	image1 = np.sum(data1[j-5:j+5,:,:],axis=0) - mean1
	image2 = np.sum(data2[i-5:i+5,:,:],axis=0) - mean2

	#Now subsample each pixel to 5x5 subpixels to better increase accuracy of alignment. 
	scalefactor = 5.
	image1 = ndimage.zoom(image1, scalefactor, order=3)
	image2 = ndimage.zoom(image2, scalefactor, order=3)
	
	#Correlate the subsampled images to find where they line up best 
	corr = ndimage.correlate(image1,image2,mode='constant')
	corrout=fits.PrimaryHDU(corr)
	corrout.writeto("corr"+str(i)+".fits",overwrite=True)

	#Optional save to file for later examination
	#corrin = fits.open("corr"+str(i)+".fits")
	#corr=corrin[0].data

	#Find pixel with largest correlation
	pos=np.unravel_index(corr.argmax(),corr.shape)
	print(pos,corr[pos[0],pos[1]])
	
	#Add position of pixel to list
	positions.append(pos)

	#Calculate center pixel of image and then add offset of max correlation pixel to list @shift
	center=[int(data1[j].shape[0]*scalefactor/2),int(data1[j].shape[1]*scalefactor/2)]
	shift.append([center[l]-pos[l] for l in range(len(pos))])
	
	#bootleg way to add point at beginning of spectral range to be used to fit translational shifts as a function of wavelength later.
	if k[0]==(0,):
		positions.append(pos)
		shift.append([center[l]-pos[l] for l in range(len(pos))])

#bootleg way to add point at beginning and end of spectral range to be used to fit translational shifts as a function of wavelength later.	
positions.append(pos)
shift.append([center[l]-pos[l] for l in range(len(pos))])
lines2=np.append(lines2,nx)
temp=np.zeros(1)
temp[0]=int((x2[0]-x0)/dx)
lines2=np.append(temp,lines2)
print(lines2)

#sanity checks
positions=np.asarray(positions)
shift=np.asarray(shift)
print(positions)
print(shift)
#print(shift[:,0],shift[:,1])
print(np.mean(shift,axis=0))

#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#ax.scatter(xs=shift[:,0],ys=shift[:,1],zs=lines2)
#ax.legend()
#plt.show()

#Fit 1d line to y-shifts across the spectral range to find shift as a function spectral range
y=interp1d(((dx*lines2)+x0),shift[:,0],kind='linear')

#sanity check
plt.figure(figsize=(10,5))
plt.plot(x2,y(x2),'b-')
plt.plot(((dx*lines2)+x0),shift[:,0],'ro')
plt.legend()
plt.show()

#Fit 1d line to z-shifts across the spectral range to find shift as a function spectral range
z=interp1d(((dx*lines2)+x0),shift[:,1],kind='linear')
plt.figure(figsize=(10,5))
plt.plot(x2,z(x2),'b-')
plt.plot(((dx*lines2)+x0),shift[:,1],'ro')
plt.legend()
plt.show()

#calculate shifts then using best fit 2D line. 
trans=[(y(x2)[i]/5.,z(x2)[i]/5.) for i in range(0,len(x2))]

#sanity check
print(trans[2741])

#only want simple translation in the affine transformation later so use identity matrix for tranformation matrix. 
trans_mat=np.asarray([[1,0],[0,1]])

#initialize shifted cubes
shiftedcube=np.zeros(np.shape(data2))
dqcube=np.zeros(np.shape(dq2))
varcube=np.zeros(np.shape(var2))
overallshift=(np.median(shift[:,0])/5.,np.median(shift[:,1])/5.)
print(overallshift)
print(len(data2),len(trans),len(data1),data1[0].shape)

#perform translation transformation for each wavelength step
for i in range(0,len(data2)):
	shiftframe=ndimage.affine_transform(data2[i],trans_mat,offset=overallshift,mode='constant',cval=0.0,output_shape=data1[0].shape)
	varshift=ndimage.affine_transform(var2[i],trans_mat,offset=overallshift,mode='constant',cval=0.0,output_shape=data1[0].shape)
	dqshift=ndimage.affine_transform(dq2[i],trans_mat,offset=overallshift,mode='constant',cval=1.0,output_shape=data1[0].shape)
	shiftedcube[i]=shiftframe
	varcube[i]=varshift
	dqcube[i]=dqshift

#save aligned datacube to file
cube2[1].data=shiftedcube
cube2[2].data=varcube
cube2[3].data=dqcube
outname=sys.argv[2].split('.')[0]+"_align.fits"
cube2.writeto(outname,overwrite=True)