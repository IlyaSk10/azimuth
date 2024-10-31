import h5py
import matplotlib.pyplot as plt
import numpy as np
from BachataClass import BachataClass
from rndflow import job
import os

full_batch=h5py.File('batch_obj.hdf5')
simple_batch=h5py.File('batch_Q55_VP2451.3827513285614_VS1158.8289985892447_Nd1_T2.5_fd400_new.hdf5')

list(full_batch['Channels'].keys())[0]

# +
sens_num=np.unique([int(s.split('_')[0]) for s in list(full_batch['Channels'].keys())])

MphS_total=[]

init_sens=list(full_batch['Channels'].keys())[0]

for pp in range(simple_batch['Channels'][init_sens]['data'].shape[0]):
    for i in range(6):
        MphS_X=np.zeros((sens_num.shape[0],simple_batch['Channels'][init_sens]['data'].shape[2]))
        MphS_Y=np.zeros((sens_num.shape[0],simple_batch['Channels'][init_sens]['data'].shape[2]))
        MphS_Z=np.zeros((sens_num.shape[0],simple_batch['Channels'][init_sens]['data'].shape[2]))
        for num,name in enumerate(sens_num):
            MphS_X[num,:]=simple_batch['Channels'][f'{name}_X']['data'][pp,i,:]
            MphS_Y[num,:]=simple_batch['Channels'][f'{name}_Y']['data'][pp,i,:]
            MphS_Z[num,:]=simple_batch['Channels'][f'{name}_Z']['data'][pp,i,:]
        MphS_total.extend([MphS_Z,MphS_X,MphS_Y])

MphS_total=np.array(MphS_total)
MphS_total=MphS_total.reshape(simple_batch['Channels'][init_sens]['data'].shape[0],6*3*sens_num.shape[0],simple_batch['Channels'][init_sens]['data'].shape[2])
MphS_total.shape

# +
MAIN={'domain_Px':[0],'domain_Py':[0],'domain_Pz':[0]}
MphS=MphS_total

#MAIN['domain_Px']=[0]
#MAIN['domain_Px']=[0]
#MAIN['domain_Px']=[0]

# k=0
# ind_v=np.zeros((len(MAIN['domain_Px'])*len(MAIN['domain_Py'])))
# for ix in range(len(MAIN['domain_Px'])):
#   for iy in range(len(MAIN['domain_Py'])):
#     ind_v[k]=len(MAIN['domain_Px'])*len(MAIN['domain_Py'])-len(MAIN['domain_Px'])*iy-ix-1  #  -1 for python
#     k+=1

k=0
ind_v=np.zeros((len(MAIN['domain_Px'])*len(MAIN['domain_Py'])))
for ix in range(len(MAIN['domain_Px'])):
  for iy in range(len(MAIN['domain_Py'])):
    ind_v[k]=len(MAIN['domain_Px'])*len(MAIN['domain_Py'])-len(MAIN['domain_Px'])*iy-ix-1  #  -1 for python
    k+=1

ind1=np.array([])
for i in range(len(MAIN['domain_Pz'])):
  ind1=np.concatenate((ind1,ind_v+len(MAIN['domain_Px'])*len(MAIN['domain_Py'])*i),axis=0)

MphS=np.take(MphS,ind1.tolist(),axis=0)



# +
indxy=np.array([])
indm=np.array([])
for i in range(1,7):
  ind2=np.array([k for k in range(int(MphS.shape[1]/6)*(i-1)+1,int(MphS.shape[1]/6)*i+1)])
  indxy=np.concatenate((indxy,ind2[0:int(len(ind2)/3)],ind2[2*int(len(ind2)/3):3*int(len(ind2)/3)],ind2[1*int(len(ind2)/3):2*int(len(ind2)/3)]),axis=0)
  indm=np.concatenate((indm,ind2[1*int(len(ind2)/3):2*int(len(ind2)/3)],ind2[2*int(len(ind2)/3):3*int(len(ind2)/3)+1]),axis=0)

MphS=MphS[:,indxy.astype(int)-1,:]
MphS[:,indm.astype(int)-1,:]=-MphS[:,indm.astype(int)-1,:]


# -

indt=np.array([2,1,3,4,-6,-5])
indt2=np.where(np.sign(indt)-1)[0]
ind=np.array([])
for i in range(len(indt)):
  ind=np.concatenate((ind,np.array([i for i in range(int((MphS.shape[1]/6))*(abs(indt[i])-1)+1,int((MphS.shape[1]/6))*abs(indt[i])+1)])),axis=0)

MphS=MphS[:,ind.astype(int)-1,:]
indt3=np.array([])
for i in range(len(indt2)):
  indt3=np.concatenate((indt3,np.array([i for i in range(int(MphS.shape[1]/6)*(indt2[i])+1,int(MphS.shape[1]/6)*(indt2[i]+1)+1)])),axis=0)

MphS[:,indt3.astype(int)-1,:]=-MphS[:,indt3.astype(int)-1,:]
power=11
file_data=MphS*10**power



# +
indexes=np.arange(len(MAIN['domain_Pz'])*len(MAIN['domain_Px'])*len(MAIN['domain_Py']))
reshape_indexes=np.reshape(indexes,[len(MAIN['domain_Pz']),len(MAIN['domain_Px']),len(MAIN['domain_Py'])])
shuffle_indexes=np.zeros((len(MAIN['domain_Pz']),len(MAIN['domain_Px']),len(MAIN['domain_Py'])),dtype=int)
for i in range(len(MAIN['domain_Pz'])):
  shuffle_indexes[i,:,:]=reshape_indexes[i,:,:].T
shuffle_indexes=np.reshape(shuffle_indexes,len(MAIN['domain_Pz'])*len(MAIN['domain_Px'])*len(MAIN['domain_Py']))
file_data=file_data[shuffle_indexes,:,:]
#--------------------------------------

file_data=file_data.T
# -

L=1000
num_components=18
num_sensors=23

# +
file_data = np.reshape(file_data, (L, num_components, num_sensors, file_data.shape[-1]))
print('batch shape',file_data.shape)

data_final=[]
for i in range(num_sensors):    
  data_sen = np.transpose(file_data[:, :, i, :])
  data_sen_Z = data_sen[:, [0, 3, 6, 9, 12, 15], :]
  data_sen_X = data_sen[:, [1, 4, 7, 10, 13, 16], :]
  data_sen_Y = data_sen[:, [2, 5, 8, 11, 14, 17], :]
  data_final.extend((data_sen_Z,data_sen_X,data_sen_Y))
# -

simple_batch.keys()

simple_batch['fd'][:]

data_final=[]
for i in range(num_sensors):    
  data_sen = np.transpose(file_data[:, :, i, :])
  data_sen_Z = data_sen[:, [0, 3, 6, 9, 12, 15], :]
  data_sen_X = data_sen[:, [1, 4, 7, 10, 13, 16], :]
  data_sen_Y = data_sen[:, [2, 5, 8, 11, 14, 17], :]
  data_final.extend((data_sen_Z,data_sen_X,data_sen_Y))

# +

fd=400
xcenter=417280.70001472736
ycenter=7205209.130729003
sensors_names=[str(i) for i in range(1,24)]
domainX=np.array([0.])
domainY=np.array([0.])
domainZ=np.array([0.])
flt=np.array([[1.00000000e+00, 4.19766550e+05, 7.20723841e+06],
       [2.00000000e+00, 4.15288801e+05, 7.20637430e+06],
       [3.00000000e+00, 4.16691947e+05, 7.20666707e+06],
       [4.00000000e+00, 4.17515566e+05, 7.20642129e+06],
       [5.00000000e+00, 4.18571356e+05, 7.20667227e+06],
       [6.00000000e+00, 4.20003915e+05, 7.20661578e+06],
       [7.00000000e+00, 4.15003618e+05, 7.20514183e+06],
       [8.00000000e+00, 4.15706881e+05, 7.20528647e+06],
       [9.00000000e+00, 4.16641177e+05, 7.20482996e+06],
       [1.00000000e+01, 4.18172941e+05, 7.20505150e+06],
       [1.10000000e+01, 4.20122003e+05, 7.20569077e+06],
       [1.20000000e+01, 4.20024092e+05, 7.20499810e+06],
       [1.30000000e+01, 4.14739768e+05, 7.20417538e+06],
       [1.40000000e+01, 4.16168723e+05, 7.20372561e+06],
       [1.50000000e+01, 4.17962615e+05, 7.20370223e+06],
       [1.60000000e+01, 4.19408284e+05, 7.20414332e+06],
       [1.70000000e+01, 4.20244100e+05, 7.20430183e+06],
       [1.80000000e+01, 4.16804758e+05, 7.20277090e+06],
       [1.90000000e+01, 4.18391225e+05, 7.20324193e+06],
       [2.00000000e+01, 4.20754905e+05, 7.20364525e+06],
       [2.10000000e+01, 4.17896417e+05, 7.20218210e+06],
       [2.20000000e+01, 4.18701038e+05, 7.20275885e+06],
       [2.30000000e+01, 4.19418163e+05, 7.20276585e+06]])

# +
filename = f'batch_upd_new.hdf5'

kwargs = {'data_type': 'full_wave_time',
            'channels': ['Z','X','Y'],
            'L': L,
            'fd': fd,
            'xcenter': xcenter,
            'ycenter': ycenter,
            #'grid_step': np.array([cellsize]*3,dtype=int),
            #'altitude': np.array([altitude],dtype=float),
            #'mult_degree': np.array([10**power]),
            'sensor': sensors_names, 
            'domainX': domainX,
            'domainY': domainY,
            'domainZ': domainZ,
            'field': flt,
            'fmin': 20, # fmin
            'fmax': 40, # fmax
            'data': data_final
            #'newton_power':[newton_power],
            #'modelling_step':[10]
            }
            
jpath=job.save_package(label='Simple batch')
fpath = str(jpath)+'/files/'
os.mkdir(fpath)

obj = BachataClass(fpath+filename,**kwargs)
# -

fpath+filename

simple_upd_batch=h5py.File(fpath+filename)

#sens='3_Y'
#tenz=0
comp=0
#plt.plot(np.sum(full_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='full sum')
for i in range(1,5):
    sens=f'{i}_X'
    fig, ax = plt.subplots(1,2)
    #ax[0].plot(np.sum(simple_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='simple sum')
    #ax[0].plot(np.sum(simple_upd_batch['Channels'][sens]['data'][0,:3,:],axis=0)/1e11,label='simple upd sum')
    #ax[0].plot(np.sum(full_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='full sum')
    ax[0].plot(simple_batch['Channels'][sens]['data'][0,comp,:],label='simple')
    ax[0].plot(simple_upd_batch['Channels'][sens]['data'][0,comp,:]/1e11,label='simple upd')
    
    ax[0].set_xlim(300,800)
    ax[0].set_title(sens)
    ax[0].legend(loc='upper left')

    #ax[1].plot(simple_batch['Channels'][sens]['data'][0,comp,:],label='simple sum')
    #ax[1].plot(simple_upd_batch['Channels'][sens]['data'][0,comp,:]/1e11,label='simple upd sum')
    ax[1].plot(full_batch['Channels'][sens]['data'][0,comp,:],label='full')

    ax[1].set_xlim(300,800)
    ax[1].set_title(sens)
    ax[1].legend(loc='upper left')

#sens='3_Y'
#tenz=0
comp=0
#plt.plot(np.sum(full_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='full sum')
for i in range(1,5):
    sens=f'{i}_X'
    fig, ax = plt.subplots()
    #ax.plot(np.sum(simple_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='simple sum')
    #ax.plot(np.sum(simple_upd_batch['Channels'][sens]['data'][0,:3,:],axis=0)/1e11,label='simple upd sum')
    #ax.plot(np.sum(full_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='full sum')

    ax.plot(simple_batch['Channels'][sens]['data'][0,comp,:],label='simple sum')
    ax.plot(simple_upd_batch['Channels'][sens]['data'][0,comp,:]/1e11,label='simple upd sum')
    ax.plot(full_batch['Channels'][sens]['data'][0,comp,:],label='full sum')

    ax.set_xlim(300,800)
    plt.title(sens)
    plt.legend()

plt.plot(np.sum(full_batch['Channels'][sens]['data'][0,:3,:],axis=0),label='full sum')


