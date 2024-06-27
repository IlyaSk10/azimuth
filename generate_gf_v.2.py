#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import glob
import re
import h5py
import green2
import os
import shutil
import math

#from BachataClass_Sakhalin import *
from BachataClass import *

from rndflow import job


def calculation_velocities(Z1,Z2,property_data):
    
    property_data_depths = np.unique(np.array([float(line.split()[0]) for line in property_data]))

    prop_data=property_data[np.abs(Z1-property_data_depths).argmin():np.abs(Z2-property_data_depths).argmin()+1]
    
    Vp,Vs=[],[]
    for i in range(len(prop_data)):
        vp,vs=list(map(lambda x: float(x),prop_data[i].split()[-2:]))
        Vp.append(vp)
        Vs.append(vs)

    Vp,Vs=np.mean(Vp),np.mean(Vs)
    
    return Vp,Vs
    
#---------------BEGIN------------------------------------------------------
# Load everything from input package:
globals().update(job.load())    #load variables from input package
globals().update(job.params())  #load parameters from input package
pkg=job.packages()[0]
fields=pkg.fields

channels=['Z']
num_channels=[2]

# modelator params

submodels_start,submodels_end,submodels_step=subbachata_range
r_min,r_max,r_step=radius_range

L=int(fd*tmod)
pkgs_names=['P','S']

print('---------------------------------------------------------------')
print('tmod', tmod)
print('fd', fd)
print('Power of coefficient at response', power)
print('fmin', fmin, 'fmax', fmax)
print('Q', Q)
print('Nd', Nd)
print('delta_z', delta_z)
print('subbachata_range', subbachata_range)
print('radius_range', radius_range)
print('half_wave_width', half_wave_width)
print('bachata_name', bachata_name)
print('---------------------------------------------------------------')

files = glob.glob('**',recursive = True)

zip_file = [name for name in files if '.zip' in name]

if len(zip_file)>0:
    try:
        with zipfile.ZipFile(*zip_file) as z:
            z.extractall(os.path.dirname(*zip_file))
            print("Extracted all")
    except:
        print("Invalid file")
        
files = glob.glob('**',recursive = True)
#--------------read data from config.txt-----------------------------------
config_path=[s for s in files if 'config.txt' in s][0]
with open(config_path, 'r') as f:
  s=f.readlines()
f.close()

config_result=[string.split('\n') for string  in s][6:]
# cell size
cellsize=int(re.findall(r'\d+',config_result[0][0])[0])
# altitude
altitude=float(config_result[3][0].split(':')[1])
# convert_points
convert_points=re.findall(r'\((.*)\)',config_result[5][0])[0].split(',')
convert_points=list(map(float,convert_points))
# u_points
u_points=re.findall(r'\((.*)\)',config_result[6][0])[0].split(',')[:2]
u_points=list(map(float,u_points))
# v_points
v_points=re.findall(r'\((.*)\)',config_result[7][0])[0].split(',')[:2]
v_points=list(map(float,v_points))
# arr_result
arr_result=np.array([u_points,v_points])
main_diag=np.diag(arr_result)
diag=np.fliplr(arr_result).diagonal()
if abs(main_diag[0])<abs(diag[0]) and abs(main_diag[1])<abs(diag[1]) and diag[0]<0:
  print('correct matrix')
  answer='correct'
elif abs(main_diag[0])>abs(diag[0]) and abs(main_diag[1])>abs(diag[1]) and main_diag[0]<0:
  print('not correct matrix')
  answer='incorrect'
else:
  print('check config matrix')
  answer='wrong matrix'
#------------------------------------------------------------------------

sensors_filename = [s for s in files if 'sensors.txt' in s][0]
with open(sensors_filename,'r') as f:
    data=f.readlines()
f.close()

sensors_depths=[]
for line in data[6:]: # 415 c 6, 413 c 8
    line=line.split()
    sensors_depths.append([int(line[0]),int(line[2])/100])

sensors_depths=np.array(sensors_depths)
sensors_depths_unique=np.unique(sensors_depths[:,1])

dt_space=1/(sensors_depths_unique[1]-sensors_depths_unique[0])

property_filename=[s for s in files if 'property.txt' in s][0]
with open(property_filename,'r') as f:
  property_data=f.readlines()
f.close()

number_sources=int(np.ceil((r_max-r_min+1)/r_step))

half_well=sensors_depths[:sensors_depths.shape[0]//2,:]
pkg_count=0

models_count=int(np.ceil((submodels_end-submodels_start+1)/submodels_step))
central_model=submodels_start+(models_count//2)*submodels_step

result=dict()

for depth_ind in range(submodels_start,submodels_end+1,submodels_step): # for submodels
    
    # PART1 create and modelling sudmodels
    depth_index=np.abs(half_well[:,0]-depth_ind).argmin()
    depth=half_well[depth_index,1]
    
    print(f'submodel central sensor name {depth_ind}, submodel central depth, m {depth}')

    # coords sources
    sources_coords=[]
    for x in range(r_min,r_max+1,r_step):
        sources_coords.append([x,0,depth])
        
    sources_coords=np.array(sources_coords)    
    
    Z1_ind,Z2_ind=depth_index-delta_z,depth_index+delta_z
    
    if (depth_index-delta_z)<0:
        Z1_ind=0
        Z2_ind=(depth_index+delta_z)+np.abs(depth_index-delta_z)
        
    if (depth_index+delta_z)>half_well.shape[0]-1:
        Z2_ind=half_well.shape[0]-1
        Z1_ind=(depth_index-delta_z)-np.abs((depth_index+delta_z)-(half_well.shape[0]-1))
        
    Z1,Z2=half_well[Z1_ind,1],half_well[Z2_ind,1]
    
    
    sensors_coords_z=half_well[Z1_ind:Z2_ind+1,1]
    
    sensors_coords_xy=np.zeros((sensors_coords_z.shape[0],2))
    sensors_coords=np.hstack((sensors_coords_xy,sensors_coords_z[:,np.newaxis]))
    
    vp,vs=calculation_velocities(Z1,Z2,property_data)
    
    Gtp_total=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    Gts_total=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    
    Gtp_total_out_der=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    Gts_total_out_der=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    
    Gtp_total_inv=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    Gts_total_inv=np.zeros((sensors_coords.shape[0],number_sources,3,6,L))
    # modelling
    for num_sr in range(number_sources):
        
        xx = sensors_coords-sources_coords[num_sr,:]
        rs = green2.response_t(xx, T=tmod, fs=fd)
        rs.make_green(Q=Q, vp=vp, vs=vs, fp=1e7, a3=1e-5, naive=False)

        Gtp = - rs.Gtp(nD=Nd)
        Gts = - rs.Gts(nD=Nd)
        
        Gtp_total_out_der[:,num_sr,:,:,:]=Gtp
        Gts_total_out_der[:,num_sr,:,:,:]=Gts

        Gtp_diff=np.diff(Gtp,axis=0)*dt_space
        Gts_diff=np.diff(Gts,axis=0)*dt_space
        
        Gtp_diff_inv=np.diff(Gtp[::-1,:,:,:],axis=0)*dt_space
        Gts_diff_inv=np.diff(Gts[::-1,:,:,:],axis=0)*dt_space
        
        Gtp_total[:(Gtp_total.shape[0]-1),num_sr,:,:,:] = Gtp_diff
        Gtp_total[Gtp_total.shape[0]-1,num_sr,:,:,:] = Gtp_diff[Gtp_diff.shape[0]-1,:,:,:]
        
        Gts_total[:(Gts_total.shape[0]-1),num_sr,:,:,:] = Gts_diff
        Gts_total[Gts_total.shape[0]-1,num_sr,:,:,:] = Gts_diff[Gts_diff.shape[0]-1,:,:,:]
        
        Gtp_total_inv[:(Gtp_total_inv.shape[0]-1),num_sr,:,:,:] = Gtp_diff_inv
        Gtp_total_inv[Gtp_total_inv.shape[0]-1,num_sr,:,:,:] = Gtp_diff_inv[Gtp_diff_inv.shape[0]-1,:,:,:]
        
        Gts_total_inv[:(Gts_total_inv.shape[0]-1),num_sr,:,:,:] = Gts_diff_inv
        Gts_total_inv[Gts_total_inv.shape[0]-1,num_sr,:,:,:] = Gts_diff_inv[Gts_diff_inv.shape[0]-1,:,:,:]
        
        Gtp_total_mean=(Gtp_total-Gtp_total_inv[::-1,:,:,:,:])/2
        Gts_total_mean=(Gts_total-Gts_total_inv[::-1,:,:,:,:])/2

    # PART2 separate sudmodels into P S submodels and write them in Bachata
    for wave in pkgs_names:

        data_list = []   # list of signals
        shifts_list = [] # list of signals shifts
        
        if wave=='P':
            v=vp
            Gtsum=Gtp_total_mean
        else:
            v=vs
            Gtsum=Gts_total_mean
        
        for s in range(Gtsum.shape[0]):
    
            sensor_data = np.zeros((number_sources, 6, 2*half_wave_width))
            shift_data = np.zeros((number_sources, 1), dtype = np.int16)
            
            for chan in num_channels:
                for source_p in range(Gtsum.shape[1]):
                    
                    dist=np.sqrt((sensors_coords[s,0]-sources_coords[source_p,0])**2+(sensors_coords[s,1]-sources_coords[source_p,1])**2+(sensors_coords[s,2]-sources_coords[source_p,2])**2)
                    tcenter=dist/v
                    wave_smpl_center=tcenter*fd
    
                    window_start = int(wave_smpl_center - half_wave_width)
                    window_end = int(wave_smpl_center + half_wave_width)
    
                    if window_end>=L:
                        sensor_data[source_p, :, :] = Gtsum[s,source_p,chan,:,-2*half_wave_width:]
                        shift_data[source_p,0] = int(L-2*half_wave_width)
                    elif window_start<=0:
                        sensor_data[source_p, :, :] = Gtsum[s,source_p,chan,:, 0:2*half_wave_width]
                        shift_data[source_p,0] = int(0)
                    else:
                        sensor_data[source_p, :, :] = Gtsum[s,source_p,chan,:, window_start:window_end]
                        shift_data[source_p,0] = window_start
    
                data_list.append(sensor_data*10**power) # add power coeff
                shifts_list.append(shift_data)

        # write P or S wave in Bachata
        filename_new_bachata = wave + "_" + "bachata_obj.hdf5"
        sensors=list(map(lambda x: str(int(x)),sensors_depths[:,0])) # sensors list
        field=np.insert(sensors_depths,1,np.zeros(sensors_depths.shape[0]),axis=1)
        domainZ=np.array([0])
        domainX=sources_coords[:,0]
        domainY=np.array([0])
        L_win=2*half_wave_width
        compnts=['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        xcenter,ycenter=0,0
        #grid_step=np.array([cellsize]*3,dtype=int)
        subModelName='mod_'+str(depth_ind)+'_'+wave
        # correct input half_well[Z1_ind:Z2_ind+1,0], but maybe BachataClass not correct and central_Z not correct in bachata_obj_sen.field.
        #Therefore half_well[Z1_ind-1:Z2_ind,0] for correct bachata_obj_sen.field
        sub_sensors=half_well[Z1_ind:Z2_ind+1,0].astype(int)
        sub_sensors = list(map(lambda x: sensors.index(str(x)), sub_sensors))
        sub_sensors = np.array(sub_sensors, dtype='int64')
        
        
        bachata_obj_sen = BachataClass(filename_new_bachata, 
                        data=data_list, shifts=shifts_list, subModelNames=subModelName, 
                        data_type='window_shift_time', 
                        L_win=L_win, L=L, fd=fd, fmin=fmin, fmax=fmax, components=compnts, 
                        sensors=sensors, sub_sensors=sub_sensors, central_Z = depth,
                        channels=channels, field=field, xcenter=xcenter, ycenter=ycenter,
                        domainZ=domainZ, domainX=domainX, domainY=domainY)
        
        bachata_obj_sen.altitude = np.array([altitude], dtype=float) # save altitude
        bachata_obj_sen.mult_degree = np.array([10**power]) # save mult_degree
        #bachata_obj_sen.grid_step = np.array(grid_step) # save grid_step
        
        window_shift_frequency_data = bachata_obj_sen.get_window_shift_frequency(sensors=list(np.asarray(sensors)[sub_sensors]), channels='all', points='all', components='all', subModelName=subModelName)
        
        bachata_obj_sen = BachataClass(filename_new_bachata, 
                    data=window_shift_frequency_data, shifts=shifts_list, subModelNames=subModelName, 
                    data_type='window_shift_frequency', 
                    L_win=L_win, L=L, fd=fd, fmin=fmin, fmax=fmax, components=compnts, 
                    sensors=sensors, sub_sensors=sub_sensors, central_Z = depth,
                    channels=channels, field=field, xcenter=xcenter, ycenter=ycenter,
                    domainZ=domainZ, domainX=domainX, domainY=domainY)
        
        bachata_obj_sen.altitude = np.array([altitude], dtype=float) # save altitude
        bachata_obj_sen.mult_degree = np.array([10**power]) # save mult_degree
        #bachata_obj_sen.grid_step = np.array(grid_step) # save grid_step
        
        if pkg_count==0:
            new_filename = "Bachata.hdf5"
            shutil.copyfile(filename_new_bachata, new_filename)
            big_bachata = BachataClass(new_filename)
        else:
            big_bachata.add_hdf5_file(filename_new_bachata)
        
        pkg_count+=1
        
    if central_model==depth_ind:
        
        for i in range(number_sources):
            fig = plt.figure(num=i,figsize=(15,15),constrained_layout=True,facecolor='white')
            #fig.subplots_adjust(hspace=0.3, wspace=0.7)
            #fig.tight_layout()
            fig.suptitle('central sensor name '+str(depth_ind)+',  central depth '+str(depth)+' m, source X dist '+str(sources_coords[i,0])+' m',horizontalalignment='center')
        
            ax = fig.add_subplot(4, 2, 1)    
            dat=np.sum(Gtp_total_mean[:,:,chan,:3,:],axis=2)[:,i,:]
            im=ax.imshow(dat,vmin=dat.min()/2,vmax=dat.max()/2)
            ax.set_title("P wave, sum(xx,yy,zz), derivative",fontsize=10)
            fig.colorbar(im)
            ax.set_aspect(5)
        
            ax = fig.add_subplot(4, 2, 2)
            dat1=Gts_total_mean[:,i,chan,4,:]
            im=ax.imshow(dat1,vmin=dat1.min()/2,vmax=dat1.max()/2)
            ax.set_title('S wave, xz, derivative',fontsize=10)
            fig.colorbar(im)
            ax.set_aspect(5)
        
            ax = fig.add_subplot(4, 2, 3)    
            fft_dat=np.abs(np.fft.fft(dat,axis=1))
            fft_dat=fft_dat[:,:(fft_dat.shape[1]//2)+1]
            fft_dat[:,0]/=dat.shape[1]
            fft_dat[:,1:]=(fft_dat[:,1:]/dat.shape[1])*2   
            im=ax.imshow(fft_dat,vmin=fft_dat.min()/2,vmax=fft_dat.max()/2)
            ax.set_title('P wave spectrum, sum(xx,yy,zz), derivative',fontsize=10)
            ax.set_xlabel('frequency, Hz')
            ax.set_xticks([i for i in range(0,fft_dat.shape[1]+1,20)],[round((i*fd)/dat.shape[1],0) for i in range(0,fft_dat.shape[1]+1,20)],rotation=90)
            fig.colorbar(im)
            ax.set_aspect(2.5)
        
            ax = fig.add_subplot(4, 2, 4)
            fft_dat1=np.abs(np.fft.fft(dat1,axis=1))
            fft_dat1=fft_dat1[:,:(fft_dat1.shape[1]//2)+1]
            fft_dat1[:,0]/=dat1.shape[1]
            fft_dat1[:,1:]=(fft_dat1[:,1:]/dat1.shape[1])*2   
            im=ax.imshow(fft_dat1,vmin=fft_dat1.min()/2,vmax=fft_dat1.max()/2)
            ax.set_title('S wave spectrum, xz, derivative',fontsize=10)
            ax.set_xlabel('frequency, Hz')
            ax.set_xticks([i for i in range(0,fft_dat1.shape[1]+1,20)],[round((i*fd)/dat1.shape[1],0) for i in range(0,fft_dat1.shape[1]+1,20)],rotation=90)
            fig.colorbar(im)
            ax.set_aspect(2.5)
            
            ax = fig.add_subplot(4, 2, 5)    
            dat=np.sum(Gtp_total_out_der[:,:,chan,:3,:],axis=2)[:,i,:]
            im=ax.imshow(dat,vmin=dat.min()/2,vmax=dat.max()/2)
            ax.set_title("P wave, sum(xx,yy,zz), no derivative",fontsize=10)
            fig.colorbar(im)
            ax.set_aspect(5)
            
            ax = fig.add_subplot(4, 2, 6)    
            dat1=Gts_total_out_der[:,i,chan,4,:]
            im=ax.imshow(dat1,vmin=dat1.min()/2,vmax=dat1.max()/2)
            ax.set_title("S wave, xz, no derivative",fontsize=10)
            fig.colorbar(im)
            ax.set_aspect(5)
            
            ax = fig.add_subplot(4, 2, 7)    
            fft_dat=np.abs(np.fft.fft(dat,axis=1))
            fft_dat=fft_dat[:,:(fft_dat.shape[1]//2)+1]
            fft_dat[:,0]/=dat.shape[1]
            fft_dat[:,1:]=(fft_dat[:,1:]/dat.shape[1])*2   
            im=ax.imshow(fft_dat,vmin=fft_dat.min()/2,vmax=fft_dat.max()/2)
            ax.set_title('P wave spectrum, sum(xx,yy,zz), no derivative',fontsize=10)
            ax.set_xlabel('frequency, Hz')
            ax.set_xticks([i for i in range(0,fft_dat.shape[1]+1,20)],[round((i*fd)/dat.shape[1],0) for i in range(0,fft_dat.shape[1]+1,20)],rotation=90)
            fig.colorbar(im)
            ax.set_aspect(2.5)
        
            ax = fig.add_subplot(4, 2, 8)
            fft_dat1=np.abs(np.fft.fft(dat1,axis=1))
            fft_dat1=fft_dat1[:,:(fft_dat1.shape[1]//2)+1]
            fft_dat1[:,0]/=dat1.shape[1]
            fft_dat1[:,1:]=(fft_dat1[:,1:]/dat1.shape[1])*2   
            im=ax.imshow(fft_dat1,vmin=fft_dat1.min()/2,vmax=fft_dat1.max()/2)
            ax.set_title('S wave spectrum, xz, no derivative',fontsize=10)
            ax.set_xlabel('frequency, Hz')
            ax.set_xticks([i for i in range(0,fft_dat1.shape[1]+1,20)],[round((i*fd)/dat1.shape[1],0) for i in range(0,fft_dat1.shape[1]+1,20)],rotation=90)
            fig.colorbar(im)
            ax.set_aspect(2.5)
            
            result.update({'fig, central_sensor_name '+str(depth_ind)+', depth '+str(depth)+' m, X_source_dist '+str(sources_coords[i,0])+' m.png':fig})
            
print('Done')

big_bachata.altitude = np.array([altitude], dtype=float) # save altitude
big_bachata.call_num = np.array([1], dtype=int)  # save number of calibration packages = constant 1
big_bachata.mult_degree = np.array([10**power]) # save mult_degree
#big_bachata.grid_step = np.array(grid_step) # save grid_step

num_subModels=math.ceil((submodels_end-submodels_start+1)/submodels_step)
#path = job.save_package(label = 'loc_system_'+bachata_name,fields = dict(num_subModels = num_subModels),files = {'Bachata.hdf5': lambda f: shutil.copy2(new_filename, f.parent)},images=result)
path = job.save_package(label = 'loc_system_'+bachata_name,fields = dict(num_subModels = num_subModels),files = {'Bachata.hdf5': lambda f: shutil.copy2(new_filename, f.parent)})

for name,fig in result.items():
    fig.savefig(str(path)+'/files/'+name+'.png')