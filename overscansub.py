import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
import math
from astropy.convolution import convolve, Box1DKernel
from scipy import ndimage
import sys
import importlib

'''
Script to subtract the overscan (in other words the readnoise) from the raw images. 
Must be done first, before running the datacubes through the pipeline as the pipeline
incorrectly handles this part. Edit filename to do other files.
'''

def main():

	#read in file
	infile=fits.open("raw/S20180408S0237.fits")
	
	#sanity check
	print(infile.info())
	
	#initialize output file and variables
	hdul=fits.HDUList()
	hdul.append(fits.PrimaryHDU(header=infile[0].header))
	i=0
	l=48
	x=np.arange(4224)
	for i in range(12):
		#grab extension of image and then open data of extension
		ext=infile[i+1]
		data=ext.data
		print(data.shape)

		#pull overscan region
		biassec_lis=ext.header['BIASSEC'].split('[')[1].split(']')[0].split(',')
		biassec=[biassec_lis[j].split(':') for j in range(len(biassec_lis))]
		
		#First take median along the shorter axis of the overscan region to remove outliers, then smooth to suppress variance in the readnoise
		overscan=convolve(np.median(data[:int(biassec[1][1]),int(biassec[0][0]):int(biassec[0][1])],axis=1),Box1DKernel(3))
		
		#sanity checks
		print(int(biassec[0][0]),int(biassec[0][1]),int(biassec[1][0]),int(biassec[1][1]))
		print(data[100,100],overscan[100])
		print(data.shape,overscan.shape)

		#data has type int16 so convert to int
		subdata=np.copy(data).astype(int)

		#subtract overscan
		subdata[:,:]=data[:,:]-overscan[:,None].astype(int)

		#sanity check plot. Originally saved all plots into pdf but now only show 1 every 10. 
		plt.figure
		plt.plot(overscan)
		# plt.plot(overnew(x))
		if int(biassec[0][1])>500:
			plt.plot(np.median(data[:,:int(biassec[0][0])],axis=1))
		if int(biassec[0][1])<500:
			plt.plot(np.median(data[:,int(biassec[0][1]):],axis=1))
		# plt.ylim(480.,550.)
		if i%10==0:
			plt.show()
		plt.close()

		#more sanity
		print(subdata[100,100])

		#convert back to int16 before saving to file
		subdata=np.int16(subdata)

		#update header info
		ext.header['OVERSEC']=('['+biassec[0][0]+':'+biassec[0][1]+',49:4224]','Section used for overscan calculation')
		ext.header['OVERRMS']=(.45,'Dummy number input to make gfreduce work for next step')
		ext.header['OVERSCAN']=(np.mean(overscan),'Overscan mean value')

		#append extention to the extended fits file object. 
		hdul.append(fits.ImageHDU(subdata,header=ext.header))
	
	#Write to file
	hdul.writeto("customsci237.fits",overwrite=True)







if __name__ == '__main__':
	main()
