#!/usr/bin/env python
# coding: utf-8
import numpy as np

import zipfile
from shutil import copyfile
import glob
import re
import h5py
import green2
import json
import os
import fnmatch

from BachataClass_upd import *


from rndflow import job

# Load everything from input package:
globals().update(job.load())    #load variables from input package
globals().update(job.params())  #load parameters from input package
pkg=job.packages()[0]
fields=pkg.fields

files = glob.glob('**',recursive = True)

zip_file = [name for name in files if '.zip' in name]


if len(zip_file)>0:
    try:
        zip_obj=zipfile.ZipFile(*zip_file)
        #name_list=[f for f in zip_obj.namelist() if 'property.txt' not in f]
        with zip_obj as z:
            #z.extractall(members=name_list)
            z.extractall()
            print("Extracted files")
        os.remove(*zip_file)
        print("Removed zip file") 
    except:
        print("Invalid zip file")


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
with open('sensors.txt','r') as f:
    sensors_data=f.readlines()


field_lst=[]
sensors_names=[]
for line in sensors_data:
    line=line.split()
    field_lst.append([int(re.findall(r'\d+',line[0])[0]),float(line[8]),float(line[7]),float(line[6])])
    sensors_names.append(str(int(re.findall(r'\d+',line[0])[0])))

sensors_coords=np.array(field_lst)

source_file=[i for i in files if 'points_' in i][0]
with open(source_file,'r') as f:
    sources_data=f.readlines()

sources_lst=[]
for line in sources_data:
    line=line.split()
    sources_lst.append([float(line[7]),float(line[8]),float(line[6])])

sources_coords=np.array(sources_lst)

y_center,x_center=np.mean(np.array(sources_coords)[:,:2],axis=0)
print(x_center,y_center)

property_file=[i for i in files if 'property' in i][0]
property_data = np.loadtxt(property_file)

# +
#all_props=property_data[np.where((property_data[:,0]==0) & (property_data[:,1]==0))[0],:]

# +
# for i in range(50):
#     print(i,property_data[i,0],property_data[i,1],property_data[i,3],property_data[i,4],property_data[i,6],property_data[i,7],property_data[i,8])
#     #print('\n')

# +
# vp,vs=[],[]
# for i in range(property_data.shape[0]):
#     youn_mod=property_data[i,6]
#     puas_mod=property_data[i,7]
#     density=property_data[i,8]
#     vp.append(((np.mean(density)/(1.741*1000))**4)*1000)
#     vs.append(np.sqrt(np.mean(youn_mod)/(2*np.mean(density)*(np.mean(puas_mod)+1))))

# vp,vs=np.mean(vp),np.mean(vs)

# +
num_cells_to_source = np.ceil(sources_coords[0,2]/cellsize).astype(int)

total_num_cells=0
delta_h=property_data[0,5]-property_data[0,2]
i=0
tp_sum=0
ts_sum=0

while (num_cells_to_source-total_num_cells)>delta_h:

    youn_mod=property_data[i,6]
    puas_mod=property_data[i,7]
    density=property_data[i,8]
    vp=((np.mean(density)/(1.741*1000))**4)*1000
    vs=np.sqrt(np.mean(youn_mod)/(2*np.mean(density)*(np.mean(puas_mod)+1)))
    
    tp_sum=tp_sum+(delta_h*cellsize)/vp
    ts_sum=ts_sum+(delta_h*cellsize)/vs

    total_num_cells=total_num_cells+delta_h
    i=i+1
    delta_h=property_data[i,5]-property_data[i,2]


print(num_cells_to_source,total_num_cells,delta_h)


youn_mod=property_data[i,6]
puas_mod=property_data[i,7]
density=property_data[i,8]
vp=((np.mean(density)/(1.741*1000))**4)*1000
vs=np.sqrt(np.mean(youn_mod)/(2*np.mean(density)*(np.mean(puas_mod)+1)))

tp_sum=tp_sum+((num_cells_to_source-total_num_cells)*cellsize)/vp
ts_sum=ts_sum+((num_cells_to_source-total_num_cells)*cellsize)/vs

vp,vs=sources_coords[0,2]/tp_sum,sources_coords[0,2]/ts_sum
print(vp,vs,vp/vs)

# +
# youn_mod=5042449033
# density=2120.335
# puas_mod=0.39293


# +
# vp=((np.mean(density)/(1.741*1000))**4)*1000
# vs=np.sqrt(np.mean(youn_mod)/(2*np.mean(density)*(np.mean(puas_mod)+1)))
# print(vp,vs,vp/vs)
# -


shape=np.unique(sources_coords[:,0]).shape[0]
X=np.sort(np.unique(sources_coords[:,0]))
Y=np.sort(np.unique(sources_coords[:,1]))
Z=np.sort(np.unique(sources_coords[:,2]))
width_x=X[shape//2]-X[0]
width_y=Y[shape//2]-Y[0]
xc = 0
domainX=np.linspace((xc-width_x),(xc+width_x),shape)
domainY=np.linspace((xc-width_y),(xc+width_y),shape)
domainZ=Z
Q=55
Nd = 1

Gt_res=np.zeros((sensors_coords.shape[0]*3,sources_coords.shape[0],6,round(fd*tmod)))
for i in range(sources_coords.shape[0]):
    
    xx=sensors_coords[:,1:]-sources_coords[i,[1,0,2]]
    #xx=sensors_coords[:,[2,1,3]]-sources_coords[i,:]
    rs = green2.response_t(xx, T=tmod, fs=fd)
    rs.make_green(Q=Q, vp=vp, vs=vs, fp=1e7, a3=1e-5, naive=False)
    Gtp = rs.Gtp(nD=Nd)
    Gts = rs.Gts(nD=Nd)
    Gtsum = -Gtp + Gts
    Gtsum=Gtsum[:,[2,0,1],:,:]
        
    for j in range(sensors_coords.shape[0]):
        Gt_res[3*j:3*(j+1),i,:,:]=Gtsum[j,:,:,:]


data_final=[]
for i in range(Gt_res.shape[0]):
    data_final.append(Gt_res[i,:,:,:])


# ----------------Initialization of bachata-object -------------------------
filename = f'batch_Q{Q}_VP{vp}_VS{vs}_Nd{Nd}_T{tmod}_fd{fd}_new.hdf5'

kwargs = {'data_type': 'full_wave_time',
            'channels': ['Z','X','Y'],
            'L': rs.N,
            'fd': fd,
            'xcenter': x_center,
            'ycenter': y_center,
            'grid_step': np.array([cellsize]*3,dtype=int),
            'altitude': np.array([altitude],dtype=float),
            'mult_degree': np.array([10**power]),
            'sensor': sensors_names, 
            'domainX': domainX,
            'domainY': domainY,
            'domainZ': domainZ,
            'field': np.array(field_lst)[:,:3],
            'fmin': rs.ff.min(), # fmin
            'fmax': rs.ff.max(), # fmax
            'data': data_final 
            }

jpath=job.save_package(label='Simple batch')
fpath = str(jpath)+'/files/'
os.mkdir(fpath)

obj = BachataClass(fpath+filename,**kwargs)

def integral(signal):
    N=signal.shape[0]
    integ=np.empty_like(signal)
    integ[0]=signal[0];
    
    for i in range(1,N):
        integ[i]=integ[i-1]+signal[i]
        #r=(signal[i]+signal[i-1])#*fd
        #integ.append(r)
    res=np.array(integ)
    return res


import matplotlib.pyplot as plt
import h5py
from scipy import signal
import glob

sensor='1_Z'
tensor_comp=2
samples=int(fd*tmod)

files = glob.glob('**',recursive = True)
path_to_full_modelator=[f for f in files if filename in f][0]
print(path_to_full_modelator)

simple_mod=h5py.File(path_to_full_modelator)
full_mod=h5py.File('batch_obj.hdf5') # power=1e4, modelling_step=0.001 sig_len=5000

newton_power=full_mod['newton_power'][0]
mult_degree=full_mod['mult_degree'][0]
modelling_step=full_mod['modelling_step'][0]
grid_step=full_mod['grid_step'][0]
fd=full_mod['fd'][0]

k=newton_power*mult_degree*modelling_step*grid_step#*(1/fd)


# simple plot
# plt.plot(simple_mod['Channels'][sensor]['data'][0,tensor_comp,:samples],label='simple modelator')
# plt.plot(full_mod['Channels'][sensor]['data'][0,tensor_comp,:samples],label='full wave modelator')

#points
x_p,X_p=4966,417280.70001472736
y_p,Y_p= 2964,7205209.130729003
z_p,Z_p=2262,2262

# +
#tdsh1 228 382 2964 4966 174 2262 7205209.130729003 417280.70001472736
# -

all_sens=np.loadtxt('sensors.txt',usecols = [i for i in range(1,9)])

#sample
ss=1
x_s,X_s=all_sens[ss-1,3],all_sens[ss-1,7]
y_s,Y_s=all_sens[ss-1,2],all_sens[ss-1,6]
z_s,Z_s=0,0

print(x_s,X_s,y_s,Y_s)

# +
# #sample
# x_s,X_s=4069,418172.9413
# y_s,Y_s=3120,7205051.503
# z_s,Z_s=0,0
# -

d_local=np.sqrt((x_p-x_s)**2+(y_p-y_s)**2+(z_p-z_s)**2)
d_global=np.sqrt((X_p-X_s)**2+(Y_p-Y_s)**2+(Z_p-Z_s)**2)
print(d_local,d_global)

# centre explosion
sum_simple_mod=np.sum(simple_mod['Channels'][sensor]['data'][0,:3,:samples],axis=0)
sum_full_mod=np.sum(full_mod['Channels'][sensor]['data'][0,:3,:samples],axis=0)

# filtration
fs=10
sos = signal.butter(10, fs, 'lowpass', fs=fd, output='sos')
#plt.plot(signal.sosfilt(sos, sum_simple_mod),label='sum, simple modelator')
plt.plot(sum_simple_mod,label='sum, simple modelator')
#plt.plot(integral(signal.sosfilt(sos, sum_full_mod))/k,label="integral(sum, full modelator)")
plt.plot(integral(sum_full_mod)/k,label="integral(sum, full modelator)")
plt.axvline((d_global/vp)*fd,c='r')
print((d_local/vp)*fd,(d_global/vp)*fd)
plt.xlabel('samples')
plt.grid()
plt.legend()
plt.title('Vp sensor '+str(sensor)+' dist '+str(d_global))

# filtration
samples_st=1000
fs=10
sos = signal.butter(10, fs, 'lowpass', fs=fd, output='sos')
filt_sig=signal.sosfilt(sos,full_mod['Channels'][sensor]['data'][0,2,:])
plt.plot(integral(filt_sig)/k,label="integral(sum, full modelator)")
plt.plot(simple_mod['Channels'][sensor]['data'][0,2,:],label='sum, simple modelator')
plt.axvline((d_global/vs)*fd,c='r')
print((d_local/vs)*fd,(d_global/vs)*fd)
plt.xlabel('samples')
plt.grid()
plt.legend()
plt.title('Vs')


