import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
#import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table, Column
import math
import numpy as np
import scipy.stats as stats
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
#import pylab as P
#from scipy import ndimage
#from stsci import imagestats
import sys
from astropy.visualization import (ImageNormalize, ZScaleInterval)
#from photutils import aperture_photometry, CircularAperture
#from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
import emcee
import corner
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

'''
Uses markov chain monte carlo fitting technique to fit either 1 or 2 components to the [OIII] doublet 
in each spatial pixel. First it tries to fit 2 components, then evaluates the S/N ratio to determine if 
there is statistical support for the existance of a second component. If not, then refit but with only 
one component. Appends the best fit parameters and uncertainties to the datacube file. 
'''

def model(x,theta):
	'''
	model with two gaussian components used for mcmc fitting of [OIII] on spaxels with broad wings
	'''
	#object redshift from redshiftcalc
	z=1.0040871

	#initialize the output variable + continuum
	y=np.zeros_like(x)+theta[-1]

	#amplitude of first component for 4959
	a1=theta[0]
	#amplitude of first component for 5007
	a2=2.94*theta[0]

	#offset in spectral direction of first component of doublet
	x01=theta[2]+4958.91*z
	x02=theta[2]+5006.84*z

	#linewidth of first component
	sig1=theta[4]

	#add first component doublet to output
	y += a1/(sig1*(2.*np.pi)**(0.5))*np.exp(-0.5*((x-x01)/sig1)**2.0)+a2/(sig1*(2.*np.pi)**(0.5))*np.exp(-0.5*((x-x02)/sig1)**2.0)
	
	#same as above but for second component now
	a1=theta[1]
	a2=2.94*theta[1]
	x01=4958.91*z+theta[3]
	x02=5006.84*z+theta[3]
	s=theta[5]

	#use @wings function for the broad component 
	y+=wings(x,a1,x01,s)+wings(x,a2,x02,s)
	return y

def model2(x,theta):
	'''
	Same as @model but only with one component, and no wing component
	'''
	z=1.0040871
	x0t=np.array((4958.91*z,5006.84*z))
	theta2=np.zeros(14)
	theta2[0],theta2[1]=theta[0],theta[0]*2.94
	theta2[2],theta2[3]=theta[1],theta[1]
	theta2[4],theta2[5]=theta[2],theta[2]
	y=np.zeros_like(x)+theta[-1]
	num=2
	for i in range(num):
		a=theta2[i]
		x0=theta2[i+num]+x0t[i]
		sig=theta2[i+2*num]
		y += a/(sig*(2.*math.pi)**(0.5))*np.exp(-0.5*((x-x0)/sig)**2.0)
	return y

def gaussian(x, a, x0, sigma):
	'''
	simple gaussian function
	'''
    return a/(sigma*(2.*math.pi)**(0.5))*np.exp(-(x-x0)**2/(2.*sigma**2))

def lorentz(x,a,x0,gamma):
	'''
	simple lorentz function
	'''
	g=(x-x0)
	return (1./np.pi)*a*(gamma/2.)/((gamma/2.)**2.+g**2.)


def logprob(theta,x,y,yerr,x1):
	"""
	logprobability function used for mcmc fitting with two line components. Loglikelihood is just classic
	sum of residuals^2/error^2
	"""
	theta2=np.zeros(7)
	theta2[:6]=theta
	theta2[6]=x1
	# print theta2.shape
	
	_lnprior = lnprior(theta2)
	if np.isfinite(_lnprior):
		_lnlike = -0.5 * np.sum((y - model(x,theta2))**2./yerr**2.)
		return _lnlike + _lnprior
	return _lnprior

def logprob2(theta,x,y,yerr,x1):
	"""
	Same as @logprob but for one line component
	"""
	theta2=np.zeros(5)
	theta2[:3]=theta
	theta2[3]=x1
	# print theta2.shape
	
	_lnprior = lnprior2(theta2)
	if np.isfinite(_lnprior):
		_lnlike = -0.5 * np.sum((y - model2(x,theta2))**2./yerr**2.)
		return _lnlike + _lnprior
	return _lnprior

def lnprior(theta):
	"""
	Priors used for 2 line components. Limits are there to remove fits that could never happen in real life. 
	The reason for the amplitude having a massive upper limit is due to scaling to remove small number effect.
	"""
	a1,x01,sig1 = theta[0],theta[2],theta[4]
	a2,x02,sig2 = theta[1],theta[3],theta[5]
	cond11 = (0<a1) & (a1<100000000)
	cond12 = (0<a2) & (a2 < 1000000000)
	cond21 = (-4<x01) & (x01<4)
	cond22 = (-4<x02) & (x02<4)
	cond31 = (0.01<sig1) & (sig1<2)
	cond32 = (8<sig2) & (sig2<50)
	cond = cond11 & cond12 & cond21 & cond31 & cond32 & cond22
	if cond:
		return 0
	return -np.Inf

def lnprior2(theta):
	"""
	Same as @lnprior but for one line component
	"""
	a,x0,sig = theta[0],theta[1],theta[2]
	cond1 = (-10<a) & (a<200000000)
	cond2 = (-8<x0) & (x0<8)
	cond3 = (0.4<sig) & (sig<2)

	cond = cond1 & cond2 & cond3
	if cond:
		return 0
	return -np.Inf

def wings(x,a,x0,gamma):
	'''
	Defines the linefunction for the wings and accounts for instrument response for the two lines in the [OIII] doublet
	'''
	gx=np.linspace(-10,10,20)
	line=lorentz(x,a,x0,gamma)
	if x[0]>5000:
		instrument_response=gaussian(gx,1.,0.0,(5006.84*1.004078)/3132.38/2.355)
		output=np.convolve(line,instrument_response,'same')
	if x[0]<5000:
		instrument_response=gaussian(gx,1.,0.0,(4958.91*1.004078)/3132.38/2.355)
		output=np.convolve(line,instrument_response,'same')

	return output

def hbcomp(xin,amp1,amp2):
	'''
	function to take the fit from [OIII] and adapt it to Hbeta. xin are the fits and amp1,2 are the amplitdues of the components scaled to hbeta
	'''
	x=np.array(xin[0])
	fixedparams=np.array(xin[1])
	x0t1,x0t2,sig1,sig2=fixedparams
	x01,x02=4880.93+x0t1,4880.93+x0t2
	return amp1/(sig1*(2.*math.pi)**(0.5))*np.exp(-(x-x01)**2./(2.*sig1**2.0))+amp2/(sig2*(2.*math.pi)**(0.5))*np.exp(-(x-x02)**2./(2.*sig2**2.0))


#get time to be used in saving the fits to a pdf for later use if needed
timestr=time.strftime("%Y%m%d-%H%M")


#Load in data, fix nans to be able to be "fit"
cube=fits.open("CombinedCube_corr.fits")
data=np.array(np.nan_to_num(cube[1].data))

#fix nans of variance array and add the error not accounted for by the pipeline
var=np.array(np.nan_to_num(cube[2].data))+(data*0.038)**2.
var[:,48,32]=var[:,47,32]

#Smooth Data in spatial dimension
smoothdata=np.zeros(np.shape(data))
smoothvar=np.zeros(np.shape(var))
kernel=Gaussian2DKernel(2.)
smoothdata=np.array([convolve(data[i,:,:],kernel) for i in range(len(data))])
smoothvar=np.array([convolve(var[i,:,:],kernel) for i in range(len(var))])
data=smoothdata
var=smoothvar

#Define wavelength arrays
z=1.0040871
nx=cube[1].header['NAXIS3']
dx=cube[1].header['CD3_3']
x0=cube[1].header['CRVAL3']
print(x0,dx,nx)
wave=np.array([dx*(l)+x0 for l in range(0,nx)])
#to be used in showing fits
x2=np.linspace(4930.,5090.,1000)
#line centers for the two components on the first line of doublet
x0t=np.array([4958.91*z,4958.91*z])

#converts wavelength space to velocity space to be used in plots
velocity2=((x2-4958.91*z)/4958.91*z)*3e5


#All plots (fits, and walkers) will be saved to a pdf
with PdfPages('Model_Comparisons'+timestr+'.pdf') as pdf:
	
	#Initial Guesses for 2 component fit
	guess = np.zeros(6)
	#amplitudes
	guess[0],guess[1]=600000.,60.
	#line offsets
	guess[2],guess[3]=0.0,0.0
	#line widths
	guess[4],guess[5]=0.5,20

	#initial guesses for 1 component fit
	guess2=np.zeros(3)
	#amplitude
	guess2[0]=60000.
	#offset
	guess2[1]=0.0
	#width
	guess2[2]=0.7
	
	#define mcmc parameters
	ndim,nwalkers,nstep=6,100,5000
	
	#initialize output cubes
	fulltruth=np.zeros((49,33,len(guess)))
	truthcube=np.zeros((ndim,49,33))*np.nan
	fiterrcube=np.zeros((ndim,49,33))*np.nan
	
	#initialize counts to be used later
	count=0
	count2=0

	#We scale up the amplitude of the spectra being fitted to remove the small number problem
	scale=100000
	
	#iterate over all spaxels
	for q in range(49):
		for w in range(33):
			#redefine if fitting went to one component
			ndim,nwalkers,nstep=6,100,5000
			#Grab spectrum of the spaxel
			y=data[:,q,w]*scale

			#make a 2D mask that will show where in the fov the spaxel is
			mask=np.zeros((49,33))
			mask[q,w]+=1
			
			#increase the uncertainty based on flux to weight the fitting routine toward the wings
			yerr=np.abs(y)**(0.5)

			#continuum value and error
			bcont=np.median(y[(4780. < wave) & (wave < 4840.)])
			contstd=np.std(y[(4780. < wave) & (wave < 4840.)])/scale
			
			#pull only [OIII] spectral range, while cutting out faint nebular lines that are not [OIII]
			xcut=wave[((4930 < wave) & (wave < 4938)) | ((4946 < wave) & (wave < 5034)) | ((5040 < wave) & (wave < 5090.))]
			xcut2=wave[(4780. < wave) & (wave < 5090.)]
			ycut=y[((4930 < wave) & (wave < 4938)) | ((4946 < wave) & (wave < 5034)) | ((5040 < wave) & (wave < 5090.))]
			ycut2=y[(4780. < wave) & (wave < 5090.)]
			yerrcut=yerr[((4930 < wave) & (wave < 4938)) | ((4946 < wave) & (wave < 5034)) | ((5040 < wave) & (wave < 5090.))]
			
			#Transform from wavelength to velocity
			velocity=((xcut-4958.91*z)/4958.91*z)*3e5
			velocitycut=((wave[(4780. < wave) & (wave < 5090.)]-4958.91*z)/4958.91*z)*3e5

			#initialize walker locations
			pos=np.array([guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])
			
			#setup mcmc object
			sampler=emcee.EnsembleSampler(nwalkers,ndim,logprob,args=[xcut,ycut,yerrcut,bcont])
			
			#burn in phase
			state=sampler.run_mcmc(pos,1500)
			
			#reset the object and then run from the end of the burn in phase
			sampler.reset()
			sampler.run_mcmc(state[0],nstep)

			#reshape to create full probability distributions for each parameter
			cornersamples=sampler.chain[:,:,:].reshape((-1,ndim))
			
			#stack them into a shape that @corner plotting can handle
			cornerdist=np.vstack((cornersamples[:,0],cornersamples[:,1],cornersamples[:,2],cornersamples[:,3],cornersamples[:,4],cornersamples[:,5])).T
			
			#Best fit parameter, use mode to remove influence of outliers
			truth=np.array([stats.mode(cornersamples[:,i])[0] for i in range(ndim)]).flatten()

			#format the best fits into shape that @corner can handle
			cornertruth=np.hstack((truth[0],truth[1],truth[2],truth[3],truth[4],truth[5]))
			
			#save to output cube
			fulltruth[q,w,:]=truth

			#plot all distributions using @corner. Shows 1-D and 2-D probability distributions
			fig=corner.corner(cornerdist,labels=['$A_1$','$A_2$','$\lambda_1$','$\lambda_2$','$\sigma_1$','$\sigma_2$'],truths=cornertruth,quantiles=[0.16,0.84],show_titles=True)
			plt.show()
			plt.close()
				
			#plot walkers, 1 every 10, to make sure they are walking
			fig2,ax=plt.subplots(ndim,1,figsize=(10,10),sharex=True)
			for k in range(ndim):
				for l in range(nwalkers)[::10]:
				    ax[k].plot(np.arange(nstep)+1,sampler.chain[l,:,k],'k-',alpha=0.5)
			plt.show()
			sys.exit()
			
			#broad line amplitude (we fit the total flux)
			bamp=truth[1]/scale/np.pi/truth[5]

			#get uncertainty of flux to evaluate overall fit
			bfiterr=np.mean([np.abs(np.percentile(cornersamples[:,1],q=84)-truth[1]),np.abs(np.percentile(cornersamples[:,1],q=16)-truth[1])])
			
			#if S/N of flux fit is better than three, keep the two component fit. 
			if (truth[1]/bfiterr > 3):
				#define narrow component lines
				n=gaussian(xcut,truth[0]/scale,(4958.91*z)+truth[2],truth[4])+gaussian(xcut,2.94*truth[0]/scale,(5006.84*z)+truth[2],truth[4])
				n2=gaussian(x2,truth[0]/scale,(4958.91*z)+truth[2],truth[4])+gaussian(x2,2.94*truth[0]/scale,(5006.84*z)+truth[2],truth[4])
				ncut=gaussian(xcut2,truth[0]/scale,(4958.91*z)+truth[2],truth[4])+gaussian(xcut2,2.94*truth[0]/scale,(5006.84*z)+truth[2],truth[4])
				#define broad component lines
				br=wings(xcut,truth[1]/scale,(4958.91*z)+truth[3],truth[5])+wings(xcut,2.94*truth[1]/scale,(5006.84*z)+truth[3],truth[5])
				br2=wings(x2,truth[1]/scale,(4958.91*z)+truth[3],truth[5])+wings(x2,2.94*truth[1]/scale,(5006.84*z)+truth[3],truth[5])
				bcut=wings(xcut2,truth[1]/scale,(4958.91*z)+truth[3],truth[5])+wings(xcut2,2.94*truth[1]/scale,(5006.84*z)+truth[3],truth[5])
				#add them up
				total=n+br+bcont/scale
				totalcut=ncut+bcut+bcont/scale
				total2=n2+br2+bcont/scale
				
				#plot them and save to pdf. This plot shows the overall line with the 2D mask showing the location of the spaxel in the fov on top, 
				#and then below a zoomed in plot that clearly shows the line.
				fig1,ax=plt.subplots(3,1,figsize=(10.82,14),gridspec_kw={'height_ratios':[2,1,1]})
				ax[0].plot(velocity2,total2,'k-',linewidth=2.0,label='total')
				ax[0].step(velocitycut,ycut2/scale,where='mid',color='b',label='data')
				ax[0].set_ylim(np.min(ycut2/scale)-.1*np.abs(np.min(ycut2/scale)),1.1*np.max(total2))
				
				#The 2-D mask, inset on overall plot
				ax2=inset_axes(ax[0],width="50%",height="70%",loc=2)
				ax2.imshow(mask,cmap=plt.cm.bwr)
				ax2.set_xticks([])
				ax2.set_yticks([])

				#zoom in plot
				ax[1].plot(velocity2,total2,'k-',linewidth=2.0,label='total')
				ax[1].step(velocitycut,ycut2/scale,where='mid',color='b',label='data')
				
				#calculating the y-limits for the zoom in
				gah=1.5*2.94*truth[1]/scale/math.pi/truth[5]+bcont/scale
				hah=3.0*contstd+bcont/scale
				mx=np.max((gah,hah))
				ax[1].set_ylim(np.min(ycut2/scale)-.1*np.abs(np.min(ycut2/scale)),(mx+bcont/scale)*1.1)

				ax[2].step(velocitycut,ycut2/scale-totalcut,where='mid',color='k')
				ax[2].set_ylabel('Residual Flux')
				for k in range(2):
					ax[k].plot(velocity2,n2+bcont/scale,'--')
					ax[k].plot(velocity2,br2+bcont/scale,'--')
				plt.legend()
				ax[2].set_xlabel('Velocity (km/s)')
				ax[0].set_ylabel('Flux')
				ax[1].set_ylabel('Flux')
				pdf.savefig(fig1)
				plt.close()
				temptruth=np.copy(truth)

				#sanity checks
				if count==0:
					print "here"
					print temptruth[0],temptruth[1],cornersamples[:10,0],cornersamples[:10,1]

				#scale back the fluxes
				temptruth[0]=temptruth[0]/scale
				temptruth[1]=temptruth[1]/scale
				cornersamples[:,0]=cornersamples[:,0]/scale
				cornersamples[:,1]=cornersamples[:,1]/scale

				#more sanity checks
				if count==0:
					print "here2"
					print temptruth[0],temptruth[1],cornersamples[:10,0],cornersamples[:10,1]

				#save outputs to output cubes
				truthcube[:,q,w]=temptruth
				fiterrcube[:,q,w]=np.array([np.mean([np.abs(np.percentile(cornersamples[:,i],q=84)-temptruth[i]),np.abs(np.percentile(cornersamples[:,i],q=16)-temptruth[i])]) for i in range(6)])
				
				#increase count
				count+=1

			#if S/N of flux is less than 3, then we refit with one component. 
			if (truth[1]/bfiterr < 3):
				#define mcmc parameters
				ndim,nwalkers,nstep=3,100,1000
				#initialize walkers
				pos=np.array([guess2 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])

				#make mcmc object
				sampler=emcee.EnsembleSampler(nwalkers,ndim,logprob2,args=[xcut,ycut,yerrcut,bcont,2])
				#burn in phase
				state=sampler.run_mcmc(pos,1500)
				sampler.reset()
				
				#run mcmc fit
				sampler.run_mcmc(state[0],nstep)

				#reshape for full distributions of each parameter
				cornersamples=sampler.chain[:,:,:].reshape((-1,ndim))
				
				#find best fit parameters
				truth=np.array([stats.mode(cornersamples[:,i])[0] for i in range(ndim)]).flatten()

				#format into proper shape for @corner
				cornerdist=np.vstack((cornersamples[:,0],cornersamples[:,1],cornersamples[:,2])).T
				cornertruth=np.hstack((truth[0],truth[1],truth[2]))

				#calculate uncertainty in fit parameters
				fiterrs=np.array([np.mean([np.abs(np.percentile(cornersamples[:,i],q=84)-truth[i]),np.abs(np.percentile(cornersamples[:,i],q=16)-truth[i])]) for i in range(3)])
				
				#define line using best fit values
				total=gaussian(xcut,truth[0]/scale,(4958.91*z)+truth[1],truth[2])+gaussian(xcut,2.94*truth[0]/scale,(5006.84*z)+truth[1],truth[2])+bcont/scale
				total2=gaussian(x2,truth[0]/scale,(4958.91*z)+truth[1],truth[2])+gaussian(x2,2.94*truth[0]/scale,(5006.84*z)+truth[1],truth[2])+bcont/scale
				totalcut=gaussian(xcut2,truth[0]/scale,(4958.91*z)+truth[1],truth[2])+gaussian(xcut2,2.94*truth[0]/scale,(5006.84*z)+truth[1],truth[2])+bcont/scale
				
				#same plot as above
				fig2,ax=plt.subplots(3,1,figsize=(10.82,14),gridspec_kw={'height_ratios':[2,1,1]})
				ax[0].plot(velocity2,total2,'k-',linewidth=2.0,label='total')
				ax[0].step(velocitycut,ycut2/scale,where='mid',color='b',label='data')
				ax[0].set_ylim(np.min(ycut2/scale)-.1*np.abs(np.min(ycut2/scale)),1.1*np.max(ycut/scale))
				
				ax2=inset_axes(ax[0],width="50%",height="70%",loc=2)
				ax2.imshow(mask,cmap=plt.cm.bwr)
				ax2.set_xticks([])
				ax2.set_yticks([])

				ax[1].plot(velocity2,total2,'k-',linewidth=2.0,label='total')
				ax[1].step(velocitycut,ycut2/scale,where='mid',color='b',label='data')
				
				gah=1.5*2.94*truth[0]/scale/math.pi/truth[2]+bcont/scale
				hah=3.0*np.std(y[(4780. < wave) & (wave < 4840.)]/scale)+bcont/scale
				mx=np.max((gah,hah))
				ax[1].set_ylim(np.min(ycut2/scale)-.1*np.abs(np.min(ycut2/scale)),mx)
				
				ax[2].step(velocitycut,ycut2/scale-totalcut,where='mid',color='k')
				ax[2].set_ylabel('Residual Flux')
				plt.legend()
				ax[2].set_xlabel('Velocity (km/s)')
				ax[0].set_ylabel('Flux')
				ax[1].set_ylabel('Flux')
				pdf.savefig(fig2)
				plt.close()

				#increase count
				count2+=1

				#Save output to ouput cubes, filling second component slots with infinities 
				truthcube[:,q,w]=np.array([truth[0]/scale,np.inf,truth[1],np.inf,truth[2],np.inf])
				fiterrcube[:,q,w]=np.array([fiterrs[0]/scale,np.inf,fiterrs[1],np.inf,fiterrs[2],np.inf])

				#just make sure all plot objects are closed. 
				plt.close()
	
#save input datacubes, and output fit parameters and uncertainty cubes to the same file 
hdul=fits.HDUList()
hdul.append(fits.PrimaryHDU(header=cube[0].header))
hdul.append(fits.ImageHDU(data,header=cube[1].header))
hdul.append(fits.ImageHDU(var,header=cube[2].header))
hdul.append(fits.ImageHDU(truthcube))
hdul.append(fits.ImageHDU(fiterrcube))
hdul.writeto("ouputfilenamehere.fits",overwrite=True)
