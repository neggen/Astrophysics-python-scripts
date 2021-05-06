import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import interpolate
from astropy.visualization import (ImageNormalize, ZScaleInterval)
from matplotlib.backends.backend_pdf import PdfPages
import sys

'''
Models the scattered light across the CCD and subtracts it. Does this by using the gaps between fiber bundles
as the data points and then interpolating those gaps over the fiber bundles themselves by fitting x-axis first, 
and then the y-axis using the x-axis fit after independently. 
'''

def remove_outliers(vector,sigma):
	'''
	Primitive way to remove outliers by using ratio of median to mean of a 2D matrix.
	If the ratio is very large or small, then the mean is markedly different than the median, 
	which implies the presence of outliers. Use this ratio, and how it varies over the second
	axis to mark rows of pixels that have outliers. Then interpolate over the row to mask them

	This is needed for the scattered light modeling because each row of pixels and column are fit and outliers will 
	disrupt the fit significantly when using curve_fit (simple least squares fitting routine)
	'''

	#finding the outliers
	findout=np.median(vector,axis=0)/np.mean(vector,axis=0)
	print(vector.shape)
	std=sigma*np.std(findout)
	ll=1-std
	ul=1+std
	mask=((findout > ul) | (findout < ll))
	
	#Manual mask based on the data I was working with. Any pixels above or below were automatically not trusted.
	mask2=(vector > 100.) | (vector < -20.)
	
	#Sanity check on how many points were being flagged as outliers
	print(str(float(np.sum(mask)+np.sum(mask2.flatten()))/findout.shape[0]*100.)+"% of points are outliers")
	
	#mask with nans
	vector[:,mask]=np.nan
	vector[mask2]=np.nan
	x=np.arange(vector.shape[1])
	
	#interpolate over nans
	f=interpolate.interp1d(x[~mask],vector[:,~mask],axis=1,bounds_error=False,fill_value="extrapolate")
	cleanvector=f(x)

	#return the cleaned vector
	return cleanvector

def main():

	#Load in file, need to change name to file desired
	flatname='customsci237'
	filein=fits.open("rg"+flatname+".fits")

	#initialize data array
	data=np.zeros((12,4176,512))
	#fill data array
	for i in range(12):
		data[i]=filein[i+1].data
	print(data.shape)


	#import the corresponding fiber-bundle gap mask that defines where the gaps are
	gaps=np.genfromtxt("blkmask_xS20180408S0238",dtype='int',unpack=True)[2:,:]
	print(gaps.shape)

	# corr=fits.open("brgS20180414S0139.fits")
	# # plt.figure
	# # plt.imshow(data[0]-corr[1].data)
	# # plt.show()
	# for i in range(len(gaps[0])):
		
	# 	for j in range(12):
	# 		# print("ccd: "+str(j))
	# 		amp=corr[j+1].data
	# 		print(amp.shape)
	# 		if j==0:
	# 			fibergap=np.median(amp[gaps[0,i]:gaps[1,i],:],axis=0)
	# 			# if i==0:
	# 			# 	plt.figure
	# 			# 	plt.imshow(amp[gaps[0,i]:gaps[1,i],:20])
	# 			# 	plt.show()
	# 			# print(fibergap[:10])
	# 		else:
	# 			fibergap=np.hstack((fibergap,np.median(amp[gaps[0,i]:gaps[1,i],:],axis=0)))
	# 	print(fibergap.shape)
	# 	plt.figure
	# 	plt.plot(np.arange(6144),fibergap)
	# 	plt.show()
	


	#initialize scattered light model array
	x=np.arange(512)
	y=np.arange(4176)
	xx,yy=np.meshgrid(x,y)
	scatteredlight=np.zeros((12,4176,512))*np.nan
	
	#Create copy of data array for later
	testdata=data[0]
	for i in range(11):
		testdata=np.hstack((testdata,data[i+1]))
	print(testdata.shape)

	#Show image before scattered light subtraction
	norm=ImageNormalize(testdata,interval=ZScaleInterval())
	plt.figure
	plt.imshow(testdata,norm=norm)
	plt.show()


	#Define order of fitting in x direction. Can be higher due to having many points with full coverage of axis
	xorder=3
	
	#xposition of any pixel to be used for fitting later
	xv=np.arange(512*4)

	#initialize variables to be filled in later
	#there are 15 gaps, 3 ccds, and each ccd has 4 amps, each with a width of 512 pixels
	polyfits=np.zeros((15,3,xorder+1))
	ccds=np.zeros((3,15,512*4))
	yerr=np.zeros((3,15,512*4))
	
	#There are 15 gaps in every image used here, initialize y position varibale for these gaps
	yv=np.zeros(15)
	
	#Save all fits to a pdf to be examined later if needed. 
	with PdfPages('guanolight_xfits.pdf') as pdf:
		#Iterate over all gaps (in this case 15)
		for i in range(len(gaps[0])):
			
			#Measure the y position of the gap we are on
			yv[i]=np.median(np.arange(gaps[1,i]-gaps[0,i])+gaps[0,i])
			
			#There are 3 CCDs next to each other in the x axis. Iterate over to fit each separately
			for j in range(3):
				
				#initialize the cleaned total CCD array
				cleanx=np.zeros((4176,512*4))*np.nan

				#4 amps per CCD, each is it's own extension in the fits file. Here we manually clean hot columns of pixels that are constant in all images 
				#and group each CCD amps together into one cleaned array. 
				for k in range(4):	
					
					#get overall amp number (0-11)
					l=j*4+k
					print("amp: "+str(l))
					
					#temporary data variable
					test=data[l]
					
					#upper and lower x indicies that this amp covers on the CCD
					ux,lx=512*(k+1),512*k

					#initialize cleaned amp data to be filled in
					clean=np.zeros((gaps[1,i]-gaps[0,i],512))*np.nan
					
					#All of these if, elif statements target hot pixels that are constant and are caught by the manual mask in @remove_outliers function. 
					#These pixels are removed first before removing outliers from other sources in the else. 
					if l==0:
						clean[:,53:]=remove_outliers(test[gaps[0,i]:gaps[1,i],53:],4)
					elif l==3:
						clean[:,:497]=remove_outliers(test[gaps[0,i]:gaps[1,i],:497],4)
					elif l==4:
						temp=test[gaps[0,i]:gaps[1,i],:]
						temp[:,239:243]=np.nan
						clean[:,16:]=remove_outliers(temp[:,16:],4)
						print(clean.shape)
					elif l==7:
						clean[:,:489]=remove_outliers(test[gaps[0,i]:gaps[1,i],:489],4)
					elif l==8:
						clean[:,17:]=remove_outliers(test[gaps[0,i]:gaps[1,i],17:],4)
					elif l==11:
						clean[:,:489]=remove_outliers(test[gaps[0,i]:gaps[1,i],:489],4)
					else:
						clean=remove_outliers(test[gaps[0,i]:gaps[1,i],:],4)

					#input the cleaned amp data into the cleaned CCD array
					cleanx[gaps[0,i]:gaps[1,i],lx:ux]=clean
				
				#condense each gap down to a single point along it's y axis extent to create a single line along x
				ynew=np.median(cleanx[gaps[0,i]:gaps[1,i],:],axis=0)
				
				#only fit to points that aren't nans
				mask=~np.isnan(ynew)				
				xfit=np.polyfit(xv[mask],ynew[mask],xorder)
				
				#save the polynomial fit parameters for later
				polyfits[i,j,:]=xfit

				#use difference from fit to actual image within the gap for error estimate
				yfit=np.poly1d(xfit)
				yerr[j,i,:]=np.ones(len(xv))*np.std(ynew[mask]-yfit(xv[mask]))

				#save fit across ccd for the y fits later
				ccds[j,i,:]=yfit(xv)
				fig,ax=plt.subplots()
				ax.plot(xv[mask],ynew[mask])
				ax.plot(xv,ynew)
				ax.plot(xv,yfit(xv))
				ax.set_ylim(-10.,30.)
				if (j==0) & (i==0):
					plt.show()
				pdf.savefig(fig)
				plt.close()

	#Set y axis fit order. Have to be careful with this as we will essentially be fitting each column of pixels using 15 points
	yorder=3

	#initialize y-axis for fitting and the resulting scattered light model
	ynew=np.arange(4176)
	modelscatlight=np.zeros((4176,512*12))

	#again saving fits in pdf
	with PdfPages('guanolight_yfits.pdf') as pdf:
		
		#iterating over 3 CCDs
		for i in range(3):
			
			#iterating over the columns in groups of 4 columns median'd together along the x axis
			for j in range(512):
				
				#initial x-position of column to be fit
				xpos=j*4+i*512*4

				#median 4 columns together to be fit
				z=np.median(ccds[i,:,j*4:(j+1)*4],axis=-1)
				
				#pull errors from x-axis fits
				err=yerr[i,:,j*4]
				# z=np.hstack((z,z[-1]))
				# print(np.median(testdata[0:10,i*512*4+j],axis=0))
				
				#fit along y-axis (sorry for poor variable naming). Weight each point in fit by the inverse of the error.
				zfit=np.polyfit(yv,z,yorder,w=err**(-1.))
				znew=np.poly1d(zfit)
				
				#save the fit for those 4 columns into the model
				modelscatlight[:,xpos:xpos+4]=np.array([znew(ynew),znew(ynew),znew(ynew),znew(ynew)]).T
				
				#plot for pdf
				fig,ax=plt.subplots()
				ax.plot(ynew,znew(ynew),'r',linewidth=2.0)
				ax.errorbar(yv,z,yerr=err,fmt='o',color='b',ms=4.0)
				pdf.savefig(fig)
				
				#show first plot for sanity check
				if (i==0) & (j==0):
					plt.show()
				plt.close()

	#plot total model after all fitting
	plt.figure
	plt.imshow(modelscatlight)
	plt.show()

	#subtract the model from the data over all 12 amps
	for i in range(12):
		filein[i+1].data=data[i]-modelscatlight[:,512*i:512*(i+1)]


	#here we examine all the fiber-bundle gaps post subtraction to see how close to zero they become. If perfectly subtracted then they should all show zero flux. 
	#iterate over each gap
	for i in range(len(gaps[0])):
		
		plt.figure

		#iterate over each amp
		for j in range(12):
			# print("ccd: "+str(j))
			amp=filein[j+1].data
			print(amp.shape)
			
			#if first time then initialize variable otherwise add new amp to variable
			if j==0:
				fibergap=np.median(amp[gaps[0,i]:gaps[1,i],:],axis=0)
			else:
				fibergap=np.hstack((fibergap,np.median(amp[gaps[0,i]:gaps[1,i],:],axis=0)))
		print(fibergap.shape)

		#plot gap spanning all amps
		plt.plot(np.arange(6144),fibergap)
		plt.show()

	#write to subtracted image to file. 
	filein.writeto("brgcustomsci237.fits",overwrite=True)


if __name__ == '__main__':
	main()
