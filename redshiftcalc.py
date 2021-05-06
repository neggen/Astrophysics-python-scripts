import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import math
import sys
from photutils import aperture_photometry, CircularAperture
from scipy.optimize import curve_fit

'''
Takes in a datacube, fits the strongest emission lines interactively, and
then uses the fits to calculate the redshift of the object with uncertainties properly propagated.
'''

def gauss_cont(x, a, x0, sigma, x1):
	#Gaussian Function with continuum
    return a*np.exp(-(x-x0)**2/(2.*sigma**2)) +x1

def two_gauss_cont(x,a1,x01,sig1,a2,x02,sig2,x1):
	#2 Gaussian Functions with continuum
	return a1*np.exp(-(x-x01)**2/(2.*sig1**2))+a2*np.exp(-(x-x02)**2/(2.*sig2**2)) +x1

def linefit(x,y,amp,mean,dev,sig):
	#fitting one line function
	popt,pcov = curve_fit(gauss_cont,x,y,p0 = np.asarray([amp,mean,dev,1.0]),sigma = sig,absolute_sigma = True)
	print(popt,pcov)
	return popt,pcov

def twolinefit(x,y,amp1,mean1,dev1,amp2,mean2,dev2,sig):
	#Fitting two lines function
	popt,pcov = curve_fit(two_gauss_cont,x,y,p0 = np.asarray([amp1,mean1,dev1,amp2,mean2,dev2,1.0]),sigma = sig,absolute_sigma = True)
	print(popt,pcov)
	return popt,pcov

if len(sys.argv) < 2:
	print("Incorrect number of arguments; please provide file name")
	sys.exit(1)

#Read in file
cube = fits.open(sys.argv[1])
data = cube[1].data

#define wavelength axis
nx = cube[1].header['NAXIS3']
dx = cube[1].header['CD3_3']
x0 = cube[1].header['CRVAL3']
print(x0,dx,nx)
wavelength = np.asarray([dx*(l)+x0 for l in range(0,nx)])

#Initialize emission lines to be used for redshift calc
linename = np.zeros(19)
#linename[19] = 3770.637
linename[17] = 3797.904
linename[18] = 3835.391
linename[0] = 3868.760
linename[1] = 3888.647
# linename[2] = 3967.470
# linename[3] = 3970.079
linename[2] = 4026.190
linename[3] = 4101.742
linename[4] = 4340.471
linename[5] = 4363.210
linename[6] = 4711.260
linename[7] = 4740.120
linename[8] = 4861.333
linename[9] = 4921.93
linename[10] = 4958.911
linename[11] = 5006.843
linename[12] = 5875.624
linename[13] = 6300.304
linename[14] = 6312.060
linename[15] = 6548.050
linename[16] = 6562.819
lines = np.round((linename*(1.0+0.004046)-x0)/dx).astype(int)

#list of hydrogen lines
hlinename = np.asarray([3,4,8,16,17,18])

#list of bright lines on the 2 good CCDs
goodccd = np.array([0,3,4,5,8,10,11])

##################################
#Fitting routine for the emission lines. The aperture photometry + fitting takes a while so ran it once and then saved the line fits. 
##################################

'''

#Performing aperture photometry to get a 1D spectrum
data = np.nan_to_num(data)
onedspec = np.zeros(len(data))
pos = (14.55,25.6125)
rad = 13.256879
aperture = CircularAperture(pos,r = rad)
onedspec = np.asarray([aperture_photometry(data[i],aperture)['aperture_sum'][0] for i in range(len(data))])
var = cube[2].data
var = np.nan_to_num(var)
error = np.asarray([aperture_photometry(var[i],aperture)['aperture_sum'][0] for i in range(len(data))])

#initialize variables
means = []
meanvar = []
flux = []
fluxvar = []

#Fit for each line
for i in range(0,len(lines)):
	
	#Define wavelength ranges used for continuum measurement
	x11 = int(lines[i]-15)
	x12 = x11+5
	x22 = int(lines[i]+15)
	x21 = x22-5

	#Check if continuum regions are good, if not then update until they are
	conti = False
	while conti == False:
		fig, ax = plt.subplots(figsize = (10,5))
		ax.plot(wavelength[x11-15:x22+15],onedspec[x11-15:x22+15],'b-')
		#plt.fill_between(wavelength[x11-15:x22+15],np.asarray(onedspec)[x11-15:x22+15]+np.sqrt(np.asarray(error))[x11-15:x22+15],np.asarray(onedspec)[x11-15:x22+15]-np.sqrt(np.asarray(error))[x11-15:x22+15],interpolate = False,alpha = .2)
		ax.axvspan(wavelength[x11],wavelength[x12], alpha = 0.2,color = 'black')
		ax.axvspan(wavelength[x21],wavelength[x22], alpha = 0.2,color = 'black')
		plt.axvline(x = float(lines[i])*dx+x0,c = 'r')
		plt.show(block = False)
		#temp = (fitt[1]-linename[i])/(linename[i])
		#diff.append(fitt[1]/linename[i] -1.0)
		usrinpt = raw_input("Are continuum regions good? (y/n): ")
		if usrinpt == "y":
			plt.close()
			conti = True
		elif usrinpt == "n":
			x11 = int(raw_input("Start of first region "+str(x11)+": "))
			x12 = int(raw_input("End of first region "+str(x12)+": "))
			x21 = int(raw_input("Start of second region "+str(x21)+": "))
			x22 = int(raw_input("End of second region "+str(x22)+": "))
			plt.close()
		elif usrinpt! = "y" and usrinpt! = "n":
			print("Use either y or n.")
			plt.close()
	
	#Start by assuming only one gaussian component. Some will have two blended together
	numlines = 1
	usramp = 50.
	usrmean = float(lines[i])*dx+x0
	usrsig = 1.0

	#Try fitting routine. If bad or incorrect, update fit parameters and fit again until a good fit is reached
	fitts = False
	while fitts == False:
		fig, ax = plt.subplots(figsize = (10,5))
		if numlines == 1:
			fitt,pcov = linefit(wavelength[x11:x22],onedspec[x11:x22],usramp,usrmean,usrsig,error[x11:x22])
			ax.plot(wavelength[x11:x22],gauss_cont(wavelength[x11:x22],*fitt),'r--',label = "gaussian")
			print(pcov[1,1])
		if numlines == 2:
			usramp1 = float(raw_input("Amplitude of line 1: "))
			usramp2 = float(raw_input("Amplitude of line 2: "))
			usrmean1 = float(raw_input("Mean of line 1: "))
			usrmean2 = float(raw_input("Mean of line 2: "))
			usrsig1 = float(raw_input("Sigma of line 1: "))
			usrsig2 = float(raw_input("Sigma of line 2: "))
			fitt,pcov = twolinefit(wavelength[x11:x22],onedspec[x11:x22],usramp1,usrmean1,usrsig1,usramp2,usrmean2,usrsig2,error[x11:x22])
			ax.plot(wavelength[x11:x22],two_gauss_cont(wavelength[x11:x22],*fitt),'r--',label = "gaussian")
		ax.plot(wavelength[x11-15:x22+15],onedspec[x11-15:x22+15],'b-')
		#plt.fill_between(wavelength[x11-15:x22+15],np.asarray(onedspec)[x11-15:x22+15]+np.sqrt(np.asarray(error))[x11-15:x22+15],np.asarray(onedspec)[x11-15:x22+15]-np.sqrt(np.asarray(error))[x11-15:x22+15],interpolate = False,alpha = .2)
		plt.show(block = False)
		usrinpt = raw_input("Is fit good? (y/n):")
		if usrinpt == "y":
			plt.close()
			fitts = True
		elif usrinpt! = "y" and usrinpt! = "n":
			print("Use either y or n.")
			plt.close()
		elif usrinpt == "n":
			numlines = int(raw_input("How many lines are present? ("+str(1)+"): "))
			if numlines == 1:
				usramp = float(raw_input("Amplitude: ("+str(usramp)+"): "))
				usrmean = float(raw_input("Mean: ("+str(usrmean)+"): "))
				usrsig = float(raw_input("Stdev: ("+str(usrsig)+"): "))

	#now do some sanity checks and append fit parameters to the list.
	cont = (np.sum(onedspec[x11:x12])+np.sum(onedspec[x21:x22]))/(np.shape(onedspec[x21:x22])[0]+np.shape(onedspec[x11:x12])[0])
	print("Continuum comparison")
	print(np.shape(wavelength[x11:x12])[0],wavelength[x11:x12],fitt[-1])
	temp = ((np.sum((wavelength[x11:x12]-wavelength[x11])*fitt[-1])+np.sum((wavelength[x21:x22]-wavelength[x21])*fitt[-1]))/(np.shape(wavelength[x11:x12])[0]+np.shape(wavelength[x21:x22])[0]))
	print(cont,temp)
	flux.append(np.sum(gauss_cont(wavelength[x11:x22],fitt[0],fitt[1],fitt[2],fitt[-1]))-cont*np.shape(onedspec[x11:x22])[0])
	#fluxvar.append(np.sum(error[x11:x22])+)
	if numlines == 1:
		fluxvar.append(np.sum(error[x11:x22])+np.sum(onedspec[x11:x22]-gauss_cont(wavelength[x11:x22],fitt[0],fitt[1],fitt[2],fitt[-1]))**2.0)
	if numlines == 2:
		fluxvar.append(np.sum(error[x11:x22])+np.sum(onedspec[x11:x22]-two_gauss_cont(wavelength[x11:x22],*fitt))**2.0)
	print("signal,noise")
	print((np.sum(onedspec[x11:x22])-cont*np.shape(onedspec[x11:x22])[0]),np.sum(error[x11:x22]))
	means.append(fitt[1])
	meanvar.append(pcov[1,1])

print(means)
print(meanvar)
print(flux)
print(fluxvar)
means = np.asarray(means)
meansig = (np.asarray(meanvar)+0.105286668009**2.0)**(0.5)
flux = np.asarray(flux)
fluxsig = (np.asarray(fluxvar))**(0.5)
'''

#These are the best fit parameters from above
means = np.asarray([3884.5225173844196, 3904.7080033032858, 4042.6717688771691, 4118.5656840204529, 4358.2111325117612, 4381.0483919998151, 4730.6943113027855, 4759.5878166293442, 4881.1393346014147, 4941.9599542421538, 4979.1673005401799, 5027.3065097084127, 5899.6319831432338, 6325.9095901708461, 6337.8013038976997, 6574.6051685912462, 6589.5803392859507, 3813.0472418757486, 3850.9333269280928])
meansig = np.asarray([5.3704491502742524e-06, 1.348858286511117e-05, 3.3624514989433668e-05, 1.3923867714114244e-07, 2.1475296453830665e-08, 9.505380798644606e-08, 4.9944527153281108e-07, 5.8815913267589653e-07, 1.7206306207135115e-09, 1.194360470248419e-06, 7.7896421081889434e-10, 3.9362022897307349e-10, 1.1915125670095904e-08, 8.9908842039233743e-07, 2.2822710906617889e-07, 9.5605273785585439e-07, 2.8157090772255222e-10, 0.0021303221591047893, 0.00049584360409933006])**(0.5)
flux=np.asarray([112.46662853243376, 45.610048242436378, 3.7765950999008346, 62.369624361024911, 117.18810035142951, 37.269183163292766, 8.9466111512370645, 5.8414151988312693, 241.88812652163358, 2.7700550530719443, 507.22607964245594, 1525.8534906714656, 23.531981874925847, 2.2970606046551572, 2.6669002328007609, 1.0751373978232213, 786.72759643172287, 0.48693931810890945, 11.485394736665782])
fluxsig = np.asarray([2.788426394782626, 1.9034547840473699, 0.34578886606521142, 0.22202195868180441, 0.14375258218417741, 0.12200043527852721, 0.0531351484939247, 0.049290391350390116, 0.30533625561460276, 0.032620027935486448, 6.4028170324151024, 143.1600840072546, 0.026649209499626729, 0.01673270230621202, 0.016587566448348901, 0.013588388635911781, 0.57282796323766716, 6.6032146566899588, 3.7593166561650864])**(0.5)
meansig = (meansig**2.0+0.105286668009**2.0)**(0.5)

sn = flux/fluxsig
# print(flux)
# print(fluxsig)

mask = np.where(sn > 50.)
print(mask)
print(means.shape,lines.shape)

resel = means/3132.3807329597
diff = means-lines

#Calculate redshift and uncertainties
z = means/linename-1.0
zsig = means/linename*((meansig/means)**2.0+(.0005/linename)**2.0)**(0.5)
hline = z[hlinename]
print(means[hlinename])
print(hline)


binss = np.arange(6)*.00002+.00402
print(binss)
print(np.median(z),np.std(z),np.sum(zsig**2.0)**(0.5))
print(np.median(z[goodccd]),np.std(z[goodccd]),np.sum(zsig[goodccd]**2.0)**(0.5))
print(z[goodccd],z)

plt.figure(figsize = (10,5))
plt.errorbar(z,resel,xerr = zsig,c = 'r',fmt = 'o',label = "all lines")
plt.errorbar(z[goodccd],resel[goodccd],xerr = zsig[goodccd],c = 'b',fmt = 'o',label = "Lines with S/N > 50")

#plt.errorbar(hline[mask],resel[hlinename][mask],xerr = zsig[hlinename][mask],c = 'r',fmt = 'o',label = "hydrogen only")
#plt.hist(z[:-3],bins = binss,color = 'b',alpha = .3)
plt.axvline(x = np.median(z),c = 'k',label = 'median')
plt.axvline(x = np.mean(z),c = 'g',label = 'mean')
plt.xlabel("Redshift")
plt.ylabel(r"Spectral Resolution Element Size ($\AA$)")
plt.legend()
plt.show()

plt.figure(figsize = (10,5))
plt.errorbar(z[mask],resel[mask],xerr = zsig[mask],c = 'b',fmt = 'o',label = "all lines")
plt.errorbar(hline[mask],resel[hlinename][mask],xerr = zsig[hlinename][mask],c = 'r',fmt = 'o',label = "hydrogen only")
#plt.hist(z[:-3],bins = binss,color = 'b',alpha = .3)
plt.axvline(x = np.median(z[mask]),c = 'k',label = 'median')
plt.axvline(x = np.mean(z[mask]),c = 'g',label = 'mean')
plt.xlabel("Redshift")
plt.ylabel(r"Spectral Resolution Element Size ($\AA$)")
plt.legend()
plt.show()
