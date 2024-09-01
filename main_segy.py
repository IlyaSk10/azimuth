#!/usr/bin/env python3

import sys
import os
import ftplib
import fnmatch
from rndflow import job, server
import glob
import requests
import json
import pandas as pd
import numpy as np
from BachataClass import *
from base_func import *
import pickle
import shutil
# Load everything from input package:
globals().update(job.load())    #load variables from input package
globals().update(job.params())  #load parameters from input package
pkg=job.packages()[0]
fields=pkg.fields

files = glob.glob('**',recursive = True)    #get files list

# +
from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
from obspy.io.segy.core import _read_segy
import numpy as np
import sys

stream = Stream()

for _i in range(3):
    # Create some random data.
    data = np.random.ranf(1000)
    data = np.require(data, dtype=np.float32)
    trace = Trace(data=data)

    # Attributes in trace.stats will overwrite everything in
    # trace.stats.segy.trace_header
    trace.stats.delta = 0.01
    # SEGY does not support microsecond precision! Any microseconds will
    # be discarded.
    trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)

    # If you want to set some additional attributes in the trace header,
    # add one and only set the attributes you want to be set. Otherwise the
    # header will be created for you with default values.
    if not hasattr(trace.stats, 'segy.trace_header'):
        trace.stats.segy = {}
    trace.stats.segy.trace_header = SEGYTraceHeader()
    trace.stats.segy.trace_header.trace_sequence_number_within_line = _i + 1
    trace.stats.segy.trace_header.receiver_group_elevation = 444

    # Add trace to stream
    stream.append(trace)

# A SEGY file has file wide headers. This can be attached to the stream
# object.  If these are not set, they will be autocreated with default
# values.
stream.stats = AttribDict()
stream.stats.textual_file_header = 'Textual Header!'
stream.stats.binary_file_header = SEGYBinaryFileHeader()
stream.stats.binary_file_header.trace_sorting_code = 5

#print "Stream object before writing..."
#print stream

stream.write("TEST.sgy", format="SEGY", data_encoding=1,
             byteorder=sys.byteorder)
#print "Stream object after writing. Will have some segy attributes..."
#print stream

#print "Reading using obspy.io.segy..."
#st1 = _read_segy("TEST.sgy")
#print (st1)

#print "Reading using obspy.core..."
st2 = read("TEST.sgy")
print("File saved successfully.")

# +
import numpy as np
from obspy.core import Stream, Trace
from obspy.io.segy.segy import SEGYFile, SEGYTraceHeader, SEGYBinaryFileHeader, SEGYTextualFileHeader

# Создание данных для трассы
npts = 1000 # Количество точек данных в трассе
sampling_rate = 0.004 # Интервал дискретизации в секундах (4 мс)
data = np.random.randn(npts) # Пример данных (случайные числа)

# Создание заголовка трассы
trace_header = SEGYTraceHeader()
trace_header.trace_sequence_number_within_line = 1
trace_header.source_coordinate_x = 123456
trace_header.source_coordinate_y = 654321
trace_header.group_coordinate_x = 123456
trace_header.group_coordinate_y = 654321
trace_header.sample_interval_in_microseconds = int(sampling_rate * 1e6)
trace_header.number_of_samples_in_this_trace = npts

# Создание трассы
trace = Trace(data=data)
trace.stats.segy = {'trace_header': trace_header}

# Создание потока (Stream) и добавление трассы
stream = Stream(traces=[trace])

# Создание бинарного заголовка файла
binary_header = SEGYBinaryFileHeader()
binary_header.data_sample_format = 5 # Формат данных (5 = IEEE 4-byte float)
binary_header.number_of_data_traces_per_ensemble = 1
binary_header.number_of_samples_per_data_trace = npts
binary_header.sample_interval_in_microseconds = int(sampling_rate * 1e6)

# Создание текстового заголовка файла

# Создание SEGY-файла
segy_file = SEGYFile()
segy_file.binary_file_header = binary_header
segy_file.textual_file_header = textual_header
segy_file.traces = stream.traces

# Сохранение SEGY-файла
output_filename = "output.segy"
segy_file.write(output_filename, format='SEGY', data_encoding=5)

print(f"SEGY файл '{output_filename}' успешно создан.")

# +
#https://github.com/obspy/obspy/blob/master/obspy/io/segy/header.py

# +
print(submodel)

model_centralZ=f[submodel]['central_Z'][:].item()
model_field=f[submodel]['field'][:]
model_sensors=f[submodel]['sensors'][:]
print(model_centralZ)
# -

model_sensors

np.argwhere(model_sensors==1000)

# +
# START
from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader#,SEGYTextualFileHeader
from obspy.io.segy.core import _read_segy
import numpy as np
import sys
import h5py

dir_name='415_P'

fpaths = [i for i in files if 'files/' in i] 

f=h5py.File(fpaths[0])
bachata_obj = BachataClass(fpaths[0])

L=f['L'][:].item()
fd=f['fd'][:].item()
channel='Z'

try:
    os.mkdir(dir_name)
except FileExistsError:
    print("dir already exists")
    
submodels_name=sorted([line for line in sorted(f.keys()) if '_P' in line])

for submodel in submodels_name[:2]:

    print(submodel)

    try:
        os.mkdir(dir_name+'/'+submodel)
    except FileExistsError:
        print("folder already exists")
    
    model_centralZ=f[submodel]['central_Z'][:].item()
    model_field=f[submodel]['field'][:]
    model_sensors=f[submodel]['sensors'][:]

    data=np.zeros((len(model_sensors),12,6,L)) # shape (sensors,12,6,L)
    
    for i,sensor in enumerate(model_sensors):
        data[i,:,:,:]=bachata_obj.get_full_wave_time(sensors=str(int(sensor)+1), channels='Z', subModelName=submodel, points='all', components='all')

    data = np.require(data, dtype=np.float32)
    
    # create SEGYs
    for s,sens in enumerate(model_sensors):
    
        try:
            os.mkdir(dir_name+'/'+submodel+'/'+'sensor_num_'+str(int(sens)))
        except FileExistsError:
            print("folder already exists")
        
        for p in range(12):
            
            stream = Stream()
            
            for comp in range(6):
                  
                trace=Trace(data[s,p,comp,:])
                
                trace.stats.sampling_rate = float(fd)
                
                stream.stats = AttribDict()
                stream.stats.binary_file_header = SEGYBinaryFileHeader()
                stream.stats.binary_file_header.number_of_samples_per_data_trace=L
                stream.stats.binary_file_header.seg_y_format_revision_number=256
    
                stream.stats.textual_file_header="Segy file contains 6 traces (6 tenzor components) ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']"
    
                
                if not hasattr(trace.stats, 'stream.trace_header'):
                   trace.stats.segy = {}
                trace.stats.segy.trace_header = SEGYTraceHeader()
                #trace.trace_sequence_number_within_segy_file=4
    
                #textual_header = SEGYTextualFileHeader()
                
                stream.append(trace)
    
            path_to_segy=dir_name+'/'+submodel+'/'+'sensor_num_'+str(int(sens))+'/'+'sensor_num_'+str(int(sens))+'_source_num_'+str(p)+'.segy'
            stream.write(path_to_segy, format="segy", data_encoding=5,byteorder=sys.byteorder)
# -



# +
from obspy.io.segy.segy import _read_segy

segy_file_path = path_to_segy

segy_file = _read_segy(segy_file_path)
binary_header = segy_file.binary_file_header
format_version = binary_header.seg_y_format_revision_number

if format_version == 256:
     print("SEGY Format Version: 1.0")
else:
     print(f"SEGY Format Version: {format_version / 256:.1f}")

segy_file.textual_file_header.decode()
# -

#fpaths = [i for i in files if fnmatch.fnmatch(i, '*'+extention)] #get file (main data container)
fpaths = [i for i in files if 'files/' in i] 
bachata_obj = BachataClass(fpaths[0])
fmin = bachata_obj.fmin
fmax = bachata_obj.fmax
filenames = []
index = 0
for location_wave in ['P','S','all']:
    if freq_min>0 and freq_max>0 and freq_min<freq_max or (freq_min==0 and freq_max==0):
        if freq_min==fmin and freq_max==fmax or (freq_min==0 and freq_max==0):   # parameters does not change the model frequency range
            print('...Choosed frequency range is the same!...')
            Green_list, shifts_list, hash_list, list_of_depths = get_model_data(bachata_obj,location_wave)

        elif freq_min<fmin or freq_max>fmax:  # if new frequency range is out of model frequency range -> error
            raise Exception(f'ERROR! Check parameters <freq_min> and <freq_max>! They have to be in range [{fmin}, {fmax}]')

        else:  # if new frequency range is correct -> extract model data without changing Bachata.hdf5
            print('......Change frequency range......')
            print(f'New frequency range: [{freq_min},{freq_max}]')
            Green_list, shifts_list, hash_list, list_of_depths = get_model_data(bachata_obj, location_wave, freq_min, freq_max)
            # change frequency range
            fmin = freq_min
            fmax = freq_max
    elif freq_min>freq_max:   # incorrect data of new frequency range -> error
        raise Exception('ERROR! Change parameters <freq_min> and <freq_max>!')
    filenames.append(f'Green_list_{location_wave}.pickle')
    filenames.append(f'list_of_depths_{location_wave}.pickle')
    filenames.append(f'shifts_list_{location_wave}.pickle')
    filenames.append(f'hash_list_{location_wave}.pickle')
    
    print(f"Save model data in local directory wave:{location_wave}")
    with open(filenames[index], 'wb') as f:
        pickle.dump(Green_list, f)
    index +=1
    with open(filenames[index], 'wb') as f:
        pickle.dump(list_of_depths, f)
    index +=1
    with open(filenames[index], 'wb') as f:
        pickle.dump(shifts_list, f)
    index +=1
    with open(filenames[index], 'wb') as f:
        pickle.dump(hash_list, f)
    index +=1
    print(f"Successfully saved to local directory wave:{location_wave}")  
    print('Green data shape of 1 subModel: ', len(Green_list[0]))
    print('Shifts shape of 1 subModel: ', len(shifts_list[0]))
    print('Type Green: ', type(Green_list[0][0][0,0,0]))
    print('Type shifts: ', type(shifts_list[0][0][0,0]))

filenames.append(fpaths[0])

# Download files to ftp
def job_save_on_ftp(ftp, outdir, filename, save_in_subdirectories):
    #change path to saving directory
    try:
        ftp.cwd(outdir)
    except ftplib.error_perm as resp:
        if str(resp) == "550 Failed to change directory.":
            print("Cannot change directory")
        else:
            raise
    if ('unified_processing' in fields) and fields['unified_processing'] and ('day' in fields):
        mk_out_dir(ftp, fields['day'])
        ftp.cwd(fields['day'])
    if save_in_subdirectories and ('subdirectory' in fields):
        mk_out_dir(ftp, fields['subdirectory'])
        ftp.cwd(fields['subdirectory'])
    
    # Get all files and directories
    try:
        ldir = ftp.nlst()
    except ftplib.error_perm as resp:
        if str(resp) == "550 No files found":
            print("No files in this directory")
        else:
            raise
    
    fname = os.path.basename(filename)
    #if fnmatch.fnmatch(filename, '*.segy'):   #To download specific files.
    print("Saving... " + fname)
        
    if not (fname in ldir):
        # Read file in binary mode
        with open(filename, "rb") as file:
            # Command for Uploading the file "STOR filename"
            ftp.storbinary(f"STOR {fname}", file)
        print("{:.2f}".format(ftp.size(fname)/1024**2)+' MB')
    else:
        if overwrite:
            print('The file with a given name already exists')
            ftp.delete(fname)
            # Read file in binary mode
            with open(filename, "rb") as file:
                # Command for Uploading the file "STOR filename"
                ftp.storbinary(f"STOR {fname}", file)
            print("{:.2f}".format(ftp.size(fname)/1024**2)+' MB')
        else:
            raise NameError('The file with a given name already exists')
            
    if save_in_subdirectories and ('subdirectory' in fields):
        ftp.sendcmd('cdup')
    if ('unified_processing' in fields) and fields['unified_processing'] and ('day' in fields):
        ftp.sendcmd('cdup')
    
    ftp.sendcmd('cdup')

# make ouput directory on the ftp server
def mk_out_dir(ftp, outname):
    
    # Get all files and directories
    try:
        ldir = ftp.nlst()
    except ftplib.error_perm as resp:
        if str(resp) == "550 No files found":
            print("No files in this directory")
        else:
            raise
    
    #make dir for output files
    if not (outname in ldir):
        ftp.mkd(outname)
    else:
        print('Warning: The directory with a given name already exists')
    #out_name = os.path.join(folder, name + '.sac')
    #print("Export " + name + "...")

#get current working data layer id
def getdatalayer(server, project_id, dfnodes_id, package_id):
    #nodes = server.get(f'/projects/{project_id}/nodes')
    #dfnodes = pd.DataFrame(nodes)
    
    for i in dfnodes_id:
        url = f'https://ias.rndflow.com/api/projects/{project_id}/nodes/{i}/packages/{package_id}'
        try:
            #res = srvr.get(f'/projects/{project_id}/nodes/{node["id"]}/packages/{pkg.id}')
            res = requests.get(url, headers=server.access_header).json()
        except requests.exceptions.HTTPError:
            print('excepted')
        else:
            les = len(res)
            if les > 1:
                return res['data_layer_id'], i

# get previous node id
def getpreviousnode(server, project_id, node_id):
    dflinks = pd.DataFrame(server.get(f'/projects/{project_id}/links'))
    
    return dflinks.at[dflinks.index[dflinks.dst_id.values == node_id][0],'src_id']
    #for link in links:
    #    if link['dst_id'] == node_id:
    #        return link['src_id']

if api_prefix is None:
    api_prefix = 'location'
ftp = ftplib.FTP(ftp_address)
ftp.login(job.secret(api_prefix+'_ftp_login'),job.secret(api_prefix+'_ftp_pw'))
#ftp.login('data','94rfdkes')

# autoreplace backslashes
ftp_dir = os.path.join(*ftp_dir.split('\\'))
# if you have to change directory on FTP server.
if not (ftp_dir == ftp.pwd()):
    ftp.cwd(ftp_dir)

if add_layer_to_outdir or add_previous_node:
    srvr = server.Server(api_key=job.secret(api_prefix+'_token'))
    project_id = job.secret(api_prefix+'_project')
    #url = f'https://ias.rndflow.com/api/projects/{project_id}/data_layers/last'
    #layer = requests.get(url)
    #outdir = outdir+'_'+layer.json()['label']
    #layer = srvr.get(f'/projects/{project_id}/data_layers/last')
    
    nodes = srvr.get(f'/projects/{project_id}/nodes')
    dfnodes = pd.DataFrame(nodes).set_index('id')
    #dfsave = dfnodes[dfnodes.label.str.contains('save_on_ftp')]
    layer_id, node_id = getdatalayer(srvr, project_id, dfnodes.index, pkg.id)
    
    if add_layer_to_outdir:
        layer = srvr.get(f'/projects/{project_id}/data_layers/{layer_id}')
        outdir = layer['label'] + '_' + outdir
        
    if add_previous_node:
        pre_node_id = getpreviousnode(srvr, project_id, node_id)
        #pre_node = dfnodes[dfnodes.index == pre_node_id].to_dict('records')[0]
        #pre_node=srvr.get(f'/projects/{project_id}/nodes/{pre_node_id}')
        outdir =  outdir + '_' + dfnodes.at[pre_node_id,'label']
outdir = outdir +'_'+ pkg.label 
mk_out_dir(ftp, outdir)
print("Save in ftp")
for fi in filenames:
    job_save_on_ftp(ftp, outdir, fi, save_in_subdirectories)

ftp.close()
fields.update({"save_model_date":ftp_dir+"\\"+outdir})
fields.update({"save_model_date_main_directory":ftp_dir})
fields.update({"save_model_date_local_directory":outdir})
new_filename = fpaths[0]
path = job.save_package(label = "model_data_" + pkg.label,fields = fields)#files = {'Bachata.hdf5': lambda f: shutil.copy2(new_filename, f.parent)}
def delete_files_with_extension(directory, extension):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            os.remove(os.path.join(directory, filename))
current_directory = os.getcwd()
print("Current directory:",current_directory)

delete_files_with_extension(current_directory, '.pickle')
