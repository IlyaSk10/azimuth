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

file_sensors = np.loadtxt("sensors 413.txt")


def get_txt_header(well):
    t = [None] * 40
    t[ 0] = f"C 1"
    t[ 1] = f"C 2 Area: Piltun-Astokhskoye"
    t[ 2] = f"C 3 Well: {well}"
    t[ 3] = f"C 4"
    t[ 4] = f"C 5"
    t[ 5] = f"C 6"
    t[ 6] = f"C 7"
    t[ 7] = f"C 8 CRS: WGS 84 / UTM zone 54N"
    t[ 8] = f"C 9"
    t[ 9] = f"C10"
    t[10] = f"C11 TVD Reference Elevation: 63.700 m above MSL"
    t[11] = f"C12 Seabed / Ground Elevation: 31.280 m below MSL"
    t[12] = f"C13"
    t[13] = f"C14 Coordinate scale factor: 100.0"
    t[14] = f"C15 Elevations & depths scale factor: 100.0"
    t[15] = f"C16 sampling rate=3000 Hz"
    t[16] = f"C17 6 traces (6 power components) ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']"
    t[17] = f"C18"
    t[18] = f"C19"
    t[19] = f"C20"
    t[20] = f"C21"
    t[21] = f"C22"
    t[22] = f"C23"
    t[23] = f"C24"
    t[24] = f"C25"
    t[25] = f"C26"
    t[26] = f"C27"
    t[27] = f"C28"
    t[28] = f"C29"
    t[29] = f"C30"
    t[30] = f"C31"
    t[31] = f"C32"
    t[32] = f"C33"
    t[33] = f"C34"
    t[34] = f"C35"
    t[35] = f"C36"
    t[36] = f"C37"
    t[37] = f"C38"
    t[38] = f"C39"
    t[39] = f"C40"
    for i in range(len(t)):
        t[i] = t[i].ljust(80, ' ')
    return ''.join(t)


# +
# START
from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict, Stats
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader, SEGYTrace  # ,SEGYTextualFileHeader
from obspy.io.segy.core import _read_segy
import numpy as np
import sys
import h5py
import re

well=413
dir_name = '413_S'

dist_sr = np.arange(10,100,8)

fpaths = [i for i in files if 'files/' in i]

f = h5py.File(fpaths[0])
bachata_obj = BachataClass(fpaths[0])

L = f['L'][:].item()
fd = f['fd'][:].item()
channel = 'Z'

try:
    os.mkdir(dir_name)
except FileExistsError:
    print("dir already exists")

submodels_name = sorted([line for line in sorted(f.keys()) if '_S' in line])

for submodel in submodels_name: # exclude

    print(submodel)

    try:
        os.mkdir(dir_name + '/' + submodel)
    except FileExistsError:
        pass

    model_sensors = f[submodel]['sensors'][:] # exclude
    num_submodel = int(re.findall(r'\d+',submodel)[0])

    data = np.zeros((len(model_sensors), 12, 6, L))  # shape (sensors,12,6,L)

    for i, sensor in enumerate(model_sensors):
        data[i, :, :, :] = bachata_obj.get_full_wave_time(sensors=str(int(sensor) + 1), channels='Z',
                                                          subModelName=submodel, points='all', components='all')

    data = np.require(data, dtype=np.float32)

    # create SEGYs
    for s, sens in enumerate(model_sensors):

        try:
            os.mkdir(dir_name + '/' + submodel + '/' + 'sensor_num_' + str(int(sens)))
        except FileExistsError:
            pass

        for p in range(12):

            stream = Stream()

            for comp in range(6):

                stream.stats = AttribDict()
                stream.stats.textual_file_header = get_txt_header(well).encode()
                stream.stats.binary_file_header = SEGYBinaryFileHeader()
                stream.stats.binary_file_header.job_identification_number = well
                stream.stats.binary_file_header.line_number = well
                stream.stats.binary_file_header.reel_number = 0 
                #stream.stats.binary_file_header.sample_interval_in_microseconds = 0.0 
                stream.stats.binary_file_header.data_sample_format_code = 1 #5
                stream.stats.binary_file_header.trace_sorting_code = 1
                stream.stats.binary_file_header.vertical_sum_code = 0
                stream.stats.binary_file_header.correlated_data_traces = 1
                stream.stats.binary_file_header.binary_gain_recovered = 2
                stream.stats.binary_file_header.amplitude_recovery_method = 1
                stream.stats.binary_file_header.measurement_system = 1
                stream.stats.binary_file_header.impulse_signal_polarity = 0
                stream.stats.binary_file_header.fixed_length_trace_flag = 1 
                stream.stats.binary_file_header.number_of_samples_per_data_trace = L
                stream.stats.binary_file_header.seg_y_format_revision_number = 0x0100
                stream.stats.binary_file_header.ensemble_fold = 0

                trace = Trace(data[s, p, comp, :])
                trace.stats = Stats()

                if not hasattr(trace.stats, 'segy.trace_header'):
                    trace.stats.segy = {}
                h = SEGYTraceHeader()

                trace.stats.delta = 0.01

                h.trace_sequence_number_within_line = comp+1
                h.trace_sequence_number_within_segy_file = comp+1
                h.original_field_record_number = 1
                h.trace_number_within_the_original_field_record = comp+1
                h.receiver_group_elevation = int(file_sensors[int(sens)-1,1]) # глубина по стволу (MD) датчика приема #+
                h.surface_elevation_at_source = int(file_sensors[num_submodel-1,1])
                h.trace_sequence_number_within_segy_file = comp+1
                h.trace_identification_code = 1
                h.data_use = 1
                h.source_depth_below_surface = 0
                h.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = dist_sr[p]
                h.source_type_orientation = 0
                h.energy_source_point_number = num_submodel #
                h.datum_elevation_at_receiver_group = 6370
                h.datum_elevation_at_source = 6370
                h.scalar_to_be_applied_to_all_elevations_and_depths = -100
                h.scalar_to_be_applied_to_all_coordinates = -100
                h.source_coordinate_x = round(file_sensors[num_submodel-1,4]*100) # longtitude *100 
                h.source_coordinate_y = round(file_sensors[num_submodel-1,5]*100)
                h.group_coordinate_x = round(file_sensors[int(sens)-1,4]*100) # coord x датчика приема
                h.group_coordinate_y = round(file_sensors[int(sens)-1,5]*100) # coord y датчика приема
                h.coordinate_units = 1
                h.delay_recording_time = 0
                h.number_of_samples_in_this_trace = L
                h.gain_type_of_field_instruments = 0
                h.instrument_gain_constant = 0
                h.instrument_early_or_initial_gain = 0
                h.trace_value_measurement_unit = 0

                trace.stats.segy.trace_header = h

                stream.append(trace)

            path_to_segy = dir_name + '/' + submodel + '/' + 'sensor_num_' + str(int(sens)) + '/' + 'sensor_num_' + str(
                int(sens)) + '_source_num_' + str(p) + '.segy'
            stream.write(path_to_segy, format="SEGY") # , data_encoding=5
# -

(212100+131760+104232+66612)*27



f[submodel]['sensors'][:].shape

# +
#996	197835	1592.67	1528.97	668765.07	5867281.35
#1046	207939	1659.85	1596.15	668827.12	5867238.39
# -

print(submodel,num_submodel,int(file_sensors[num_submodel-1,1]),round(file_sensors[num_submodel-1,4]*100),int(file_sensors[num_submodel-1,5]*100))
print(sens,int(file_sensors[int(sens)-1,1]),int(file_sensors[int(sens)-1,4]*100),int(file_sensors[int(sens)-1,5]*100))

import shutil
shutil.make_archive('413_S', 'zip', '413_S')

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
