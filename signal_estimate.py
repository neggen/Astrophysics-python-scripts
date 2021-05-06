import os
import sys
import pylab as P
import numpy as np
from pyrith import *
import glob
import pyfits as pf 


if len(sys.argv) != 3:
    print("Incorrect number of arguments.")
    sys.exit(1)

#specify the exposure time in the last argument 
exposure_time = sys.argv[2]
directoryname_list = sys.argv[1:len(sys.argv)-1]
print("list is",directoryname_list)

#initialize variables
gain = 2.02
results = exposure_time+"signal_estimate.dat"
name_array = []
mean_array = []
sd_array = []
median_array = []
diff = False
scal = False

#looking through each directory
for dir in directoryname_list:
    print("First argument is", sys.argv[1])

    #change directory to each directory in argument
    os.chdir(dir)

    #create a list of all the led fits files we are interested in
    initial_img_list = np.asarray(glob.glob("*exp_"+exposure_time+"_t00*_940nm_led.*.1.1.fits")) 
    final_img_list = np.asarray(glob.glob("*exp_"+exposure_time+"_t00*_940nm_led.*.1.103.fits"))
    
    print initial_img_list
    print final_img_list

    
    name = initial_img_list[0].split('.')[0]
    initial_list = name+"_initial.lis"
    final_list = name+"_final.lis"
    sub_list = name+"_sub_image.lis"
    gain_sub_list = name+"_gain_sub_image.lis"
    
    np.savetxt(initial_list,initial_img_list,fmt='%s')
    np.savetxt(name+"_final.lis",final_img_list,fmt='%s')
    os.system("awk '{gsub(/.fits/,\".diff.fits\");print$0}' "+initial_list+" > "+sub_list)
    os.system("awk '{gsub(/.fits/,\".diff.e.fits\");print$0}' "+initial_list+" > "+gain_sub_list)
    
    
    #subtracting the first image from the last
    if diff:
        pyrith_lists(name+"_final.lis",'-',name+"_initial.lis",sub_list,0,0)
        #multiplying the image by the gain
    if scal:
        pyrith_scal(sub_list,'*',gain,gain_sub_list,0)
            
    img_list = open(gain_sub_list).read().splitlines()
    for i in img_list:
        imagehdu = pf.open(i)
        image_array= imagehdu[0].data
        imagehdu.close()
        name = i.split('.')[0]
        print(name)
        name_array = np.append(name_array,name)
        mean_array = np.append(mean_array,np.mean(image_array,axis=0))
        sd_array = np.append(sd_array,np.std(image_array,ddof=1))
        median_array = np.append(median_array,np.median(image_array))
        #append all info to their respective arrays
        
        

    #combine 4 arrays and write them to a textfile
    np.savetxt(results,zip(name_array,mean_array,median_array,sd_array),fmt='%20s')
