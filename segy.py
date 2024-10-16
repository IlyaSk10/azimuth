# START
from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict, Stats
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader, SEGYTrace  # ,SEGYTextualFileHeader
from obspy.io.segy.core import _read_segy
import numpy as np
import sys
import h5py

dir_name = '415_P'

# fpaths = [i for i in files if 'files/' in i]
#
# f = h5py.File(fpaths[0])
# bachata_obj = BachataClass(fpaths[0])
#
# L = f['L'][:].item()
# fd = f['fd'][:].item()
# channel = 'Z'

# try:
#     os.mkdir(dir_name)
# except FileExistsError:
#     print("dir already exists")

from obspy.io.segy.segy import _read_segy

segy_file = _read_segy("C:/Users/ias/PycharmProjects/azimuth/SEIC_WELL413_DASVSP_10_09_22duty_change.segy")

first_trace_header = segy_file.traces[0].header

submodels_name = sorted([line for line in sorted(f.keys()) if '_P' in line])

for submodel in submodels_name[:1]:

    print(submodel)

    try:
        os.mkdir(dir_name + '/' + submodel)
    except FileExistsError:
        # print("folder already exists")
        pass

    model_centralZ = f[submodel]['central_Z'][:].item()
    model_field = f[submodel]['field'][:]
    model_sensors = f[submodel]['sensors'][:]

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
            # print("folder already exists")
            pass

        for p in range(12):

            stream = Stream()

            for comp in range(6):

                stream.stats = AttribDict()
                stream.stats.binary_file_header = SEGYBinaryFileHeader()
                stream.stats.binary_file_header.number_of_samples_per_data_trace = L
                stream.stats.binary_file_header.seg_y_format_revision_number = 256

                stream.stats.textual_file_header = "Segy file contains 6 traces (6 tenzor components) ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'], fd=3000"

                # if not hasattr(trace.stats, 'stream.trace_header'):

                # trace.stats.segy.trace_header = SEGYTraceHeader()
                # trace_header = SEGYTraceHeader()
                # trace_header.sampling_rate = float(fd)
                # trace.stats.sampling_rate = float(fd)
                # trace.receiver_group_elevation=file_sensors[np.argwhere(file_sensors[:,0]==sens).item()][2]
                trace = Trace(data[s, p, comp, :])
                trace.stats = Stats()

                if not hasattr(trace.stats, 'segy.trace_header'):
                    trace.stats.segy = {}
                h = SEGYTraceHeader()

                trace.stats.sampling_rate = float(fd)
                # trace.stats.delta = float(fd)

                # trace.receiver_group_elevation=11#file_sensors[np.argwhere(file_sensors[:,0]==sens).item()][2]
                # trace = SEGYTrace(header=trace_header, data=data[s,p,comp,:])
                # textual_header = SEGYTextualFileHeader()
                # h.trace_sequence_number_within_line = 111
                # h.receiver_group_elevation=file_sensors[np.argwhere(file_sensors[:,0]==sens).item()][2]
                h.receiver_group_elevation = int(sens)

                trace.stats.segy.trace_header = h

                stream.append(trace)

            path_to_segy = dir_name + '/' + submodel + '/' + 'sensor_num_' + str(int(sens)) + '/' + 'sensor_num_' + str(
                int(sens)) + '_source_num_' + str(p) + '.segy'
            stream.write(path_to_segy, format="segy", data_encoding=5)