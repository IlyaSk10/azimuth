"""
BachataClass from Location.3.0.
"""
import numpy as np
from math import floor
import h5py


class BachataClass:
    """ Класс для хранения бачаты во временном или частотном представлении
    """
    __slots__ = 'filename'

    def __init__(self, filename, data='data_from_file', **kwargs):
        """Инициализация сигнала во времени:
        obj = BachataClass(filename, data=[full_data[:, 0:6, :]], data_type='full_wave_time', L=L, L_win=L_win,
                           shifts=[shifts[:, 0:6]], fmin=fmin, fmax=fmax, fd=fd, field=field, sensor='sen1',
                           channels=['Z'], domainX=domainX, domainY=domainY, domainZ=domainZ)

        :param filename: ссылка на hdf5 файл (на существующий, или для его создания)
        :param data: сигнал np.array , форма (N_points, N_components, L), либо список list [data1, data2, ...] - по
        количеству каналов
            где N_points - количество точек визуализации, N_components - количество компонент, L - длина сигнала,

        Все остальные параметры загружаются с помощью именованных аргументов **kwargs:
        shifts: смещения оклон для обрезанного сигнала, форма: (N_points, N_components), либо список list
        [shifts1, shifts2, ...] - по количеству каналов
        field: список из Xcoord, Ycoord, Altitude - для сенсора, например [165535, 556568, 200]

        data_type: тип сигнала, необходимо указать один из:
            full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency
        sensor: имя сенсора загружаемых данных, str: 'sen1'
        channels: имена каналов загружаемых данных, список строк: ['Z', 'X', 'Y']
        components: список компонент тензора, которые присутствуют в данных в правильном порядке, например:
            components = ['xz', 'xx'], по умолчанию ='all' - 6 компонент ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        logi: список точек визуализации, например [1, 2, 5, 7]. По умолчанию  ='all' и генерируется как:
            logi = [i for i in range(data.shape[0])]
        L: длина полного сигнала
        L_win: длина окна для обрезанного сигнала
        fmin: минимальная частота в частотном представлении
        fmax: максимальная частота в частотном представлении
        fd: частота дискретизации сигнала
        domainX: - сетка локальных координат точек визуализации
        xcenter: - глобальные координаты цента
        Ycenter: - глобальные координаты цента

        Все параметры являются вычисляемыми свойствами класса, при обращении к ним происходит считывание из файла.
        Например, ClassObject.L - возвращает длину сигнала
        Не изменяйте их, если не уверены в своих действиях.
        """

        self.filename = filename
        if not isinstance(data, str):
            self.create_hdf5(data, kwargs)

    def create_hdf5(self, data, kwargs):
        with h5py.File(self.filename, 'w') as f:
            if not isinstance(data, list):
                data = [data]
            if kwargs['data_type'] in ['full_wave_time', 'full_wave_frequency', 'window_shift_time',
                                       'window_shift_frequency']:
                f.create_dataset('data_type', data=[kwargs['data_type']])
            else:
                raise Exception(f"Incorrect value of data_type: {kwargs['data_type']}, you must pass one of these: "
                                f"full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency")
            if 'logi' in kwargs and not isinstance(kwargs['logi'], str):
                f.create_dataset('logi', data=kwargs['logi'])
            else:
                logi = [i for i in range(data[0].shape[0])]
                f.create_dataset('logi', data=logi)
            if 'components' in kwargs and isinstance(kwargs['components'], list):
                f.create_dataset('components', data=kwargs['components'])
            elif 'components' in kwargs and kwargs['components'] != 'all':
                f.create_dataset('components', data=[kwargs['components']])
            else:
                f.create_dataset('components', data=['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])

            if 'domainX' in kwargs:
                f.create_dataset('domainX', data=kwargs['domainX'])
            if 'domainY' in kwargs:
                f.create_dataset('domainY', data=kwargs['domainY'])
            if 'domainZ' in kwargs:
                f.create_dataset('domainZ', data=kwargs['domainZ'])

            if isinstance(kwargs['channels'], list):
                f.create_dataset('channels', data=kwargs['channels'], maxshape=(None,), chunks=True)
            else:
                f.create_dataset('channels', data=[kwargs['channels']], maxshape=(None,), chunks=True)
            if isinstance(kwargs['sensor'], list):
                f.create_dataset('sensors', data=kwargs['sensor'], maxshape=(None,), chunks=True)
            else:
                f.create_dataset('sensors', data=[kwargs['sensor']], maxshape=(None,), chunks=True)

            self.load_attributes(kwargs)
            field = f.create_dataset('field', (0, 5), maxshape=(None, 5), dtype='i8')
            channels_group = f.create_group('Channels')
            data_count = 0
            for sen_count, sen_name in enumerate(self.sensors):
                for chan_count, chan_name in enumerate(self.channels):
                    channel_g = channels_group.create_group(str(sen_name) + '_' + str(chan_name))
                    channel_g.create_dataset('data', data=data[data_count], dtype=data[data_count].dtype)
                    field.resize((field.shape[0] + 1, field.shape[1]))
                    # if len(kwargs['field'].shape)>1:
                    field[-1, :] = np.concatenate(([sen_count, chan_count], kwargs['field'][sen_count]))
                    # else:
                    #    field[-1, :] = np.concatenate(([0, data_count], kwargs['field']))
                    if 'shifts' in kwargs:
                        if not isinstance(kwargs['shifts'], list):
                            kwargs['shifts'] = [kwargs['shifts']]
                        channel_g.create_dataset('shifts', data=kwargs['shifts'][data_count], dtype='int16')
                    data_count += 1

    def load_attributes(self, kwargs):
        if 'L' in kwargs:
            self.L = kwargs['L']
        if 'L_win' in kwargs:
            self.L_win = kwargs['L_win']
        if 'fmin' in kwargs:
            self.fmin = kwargs['fmin']
        if 'fmax' in kwargs:
            self.fmax = kwargs['fmax']
        if 'fd' in kwargs:
            self.fd = kwargs['fd']
        if 'xcenter' in kwargs:
            self.xcenter = kwargs['xcenter']
        if 'ycenter' in kwargs:
            self.ycenter = kwargs['ycenter']

    @staticmethod
    def set_parameter(filename, key, value):
        with h5py.File(filename, 'a') as f:
            if key in f.keys():
                f[key][:] = value
            else:
                f.create_dataset(key, data=value)

    @staticmethod
    def get_parameter(filename, key):
        with h5py.File(filename, 'r') as f:
            if key in f.keys():
                return f[key][:]
            else:
                raise Exception(f'No {key} in hdf5 file ({filename})!')

    @property
    def L(self):
        return self.get_parameter(self.filename, 'L')[0]

    @L.setter
    def L(self, value):
        self.set_parameter(self.filename, 'L', [value])

    @property
    def L_win(self):
        return self.get_parameter(self.filename, 'L_win')[0]

    @L_win.setter
    def L_win(self, value):
        self.set_parameter(self.filename, 'L_win', [value])

    @property
    def fmin(self):
        return self.get_parameter(self.filename, 'fmin')[0]

    @fmin.setter
    def fmin(self, value):
        self.set_parameter(self.filename, 'fmin', [value])

    @property
    def fmax(self):
        return self.get_parameter(self.filename, 'fmax')[0]

    @fmax.setter
    def fmax(self, value):
        self.set_parameter(self.filename, 'fmax', [value])

    @property
    def fd(self):
        return self.get_parameter(self.filename, 'fd')[0]

    @fd.setter
    def fd(self, value):
        self.set_parameter(self.filename, 'fd', [value])

    @property
    def xcenter(self):
        return self.get_parameter(self.filename, 'xcenter')[0]

    @xcenter.setter
    def xcenter(self, value):
        self.set_parameter(self.filename, 'xcenter', [value])

    @property
    def ycenter(self):
        return self.get_parameter(self.filename, 'ycenter')[0]

    @ycenter.setter
    def ycenter(self, value):
        self.set_parameter(self.filename, 'ycenter', [value])

    @property
    def data_type(self):
        return self.get_parameter(self.filename, 'data_type')[0].decode("utf-8")

    @data_type.setter
    def data_type(self, value):
        self.set_parameter(self.filename, 'data_type', [value])

    @property
    def components(self):
        return list(map(lambda x: x.decode("utf-8"), self.get_parameter(self.filename, 'components')))

    @components.setter
    def components(self, value):
        if not isinstance(value, list):
            value = [value]
        self.set_parameter(self.filename, 'components', value)

    @property
    def logi(self):
        logi = self.get_parameter(self.filename, 'logi')
        if isinstance(logi[0], str):
            return list(map(lambda x: x.decode("utf-8"), logi))
        elif isinstance(logi, np.ndarray):
            return list(logi)
        else:
            return logi

    @logi.setter
    def logi(self, value):
        if not isinstance(value, list) and not isinstance(value, np.ndarray):
            value = [value]
        self.set_parameter(self.filename, 'logi', value)

    @property
    def field(self):
        return self.get_parameter(self.filename, 'field')

    @field.setter
    def field(self, value):
        self.set_parameter(self.filename, 'field', value)

    @property
    def channels(self):
        channels = self.get_parameter(self.filename, 'channels')
        return list(map(lambda x: x.decode("utf-8"), channels))

    @channels.setter
    def channels(self, value):
        if not isinstance(value, list):
            value = [value]
        if len(value) > len(self.channels):
            with h5py.File(self.filename, 'a') as f:
                channels_names_hdf5 = f['channels']
                channels_names_hdf5.resize((len(value),))
        self.set_parameter(self.filename, 'channels', value)

    @property
    def sensors(self):
        sensors = self.get_parameter(self.filename, 'sensors')
        return list(map(lambda x: x.decode("utf-8"), sensors))

    @sensors.setter
    def sensors(self, value):
        if not isinstance(value, list):
            value = [value]
        if len(value) > len(self.sensors):
            with h5py.File(self.filename, 'a') as f:
                sensors_names_hdf5 = f['sensors']
                sensors_names_hdf5.resize((len(value),))
        self.set_parameter(self.filename, 'sensors', value)

    @property
    def domainX(self):
        return self.get_parameter(self.filename, 'domainX')

    @domainX.setter
    def domainX(self, value):
        self.set_parameter(self.filename, 'domainX', value)

    @property
    def domainY(self):
        return self.get_parameter(self.filename, 'domainY')

    @domainY.setter
    def domainY(self, value):
        self.set_parameter(self.filename, 'domainY', value)

    @property
    def domainZ(self):
        return self.get_parameter(self.filename, 'domainZ')

    @domainZ.setter
    def domainZ(self, value):
        self.set_parameter(self.filename, 'domainZ', value)

    @property
    def powxy(self):
        return self.get_parameter(self.filename, 'powxy')

    @powxy.setter
    def powxy(self, value):
        self.set_parameter(self.filename, 'powxy', value)

    @property
    def powz(self):
        return self.get_parameter(self.filename, 'powz')

    @powz.setter
    def powz(self, value):
        self.set_parameter(self.filename, 'powz', value)

    @property
    def impact_force(self):
        return self.get_parameter(self.filename, 'impact_force')

    @impact_force.setter
    def impact_force(self, value):
        self.set_parameter(self.filename, 'impact_force', value)

    @property
    def altitude(self):
        return self.get_parameter(self.filename, 'altitude')

    @altitude.setter
    def altitude(self, value):
        self.set_parameter(self.filename, 'altitude', value)

    @property  ## number of calibration packages for location system
    def call_num(self):
        return self.get_parameter(self.filename, 'call_num')

    @call_num.setter
    def call_num(self, value):
        self.set_parameter(self.filename, 'call_num', value)

    @property  ## multiplication degree for magnitude calculation
    def mult_degree(self):
        return self.get_parameter(self.filename, 'mult_degree')

    @mult_degree.setter
    def mult_degree(self, value):
        self.set_parameter(self.filename, 'mult_degree', value)

    @property
    def answer(self):
        return self.get_parameter(self.filename, 'answer')

    @answer.setter
    def answer(self, value):
        self.set_parameter(self.filename, 'answer', value)

    @property  ## size of model's cell
    def grid_step(self):
        return self.get_parameter(self.filename, 'grid_step')

    @grid_step.setter
    def grid_step(self, value):
        self.set_parameter(self.filename, 'grid_step', value)

    @property
    def data_dtype(self):
        if self.data_type in ['full_wave_time', 'window_shift_time']:
            return 'float'
        else:
            return 'complex64'

    def is_domain(self):
        with h5py.File(self.filename, 'r') as f:
            return {'domainX', 'domainY', 'domainZ'}.issubset(f.keys())

    def is_xycenter(self):
        with h5py.File(self.filename, 'r') as f:
            return {'xcenter', 'ycenter'}.issubset(f.keys())

    def is_shifts(self, sensor, channel):
        with h5py.File(self.filename, 'r') as f:
            cannel_group = f['Channels/' + str(sensor) + '_' + str(channel)]
            return 'shifts' in cannel_group.keys()

    def get_shifts(self, sensor, channel):
        if self.is_shifts(sensor, channel):
            with h5py.File(self.filename, 'r') as f:
                return f['Channels/' + str(sensor) + '_' + str(channel) + '/shifts'][:]
        else:
            raise Exception(f"No shifts for Sensor {sensor} channel {channel}")

    def set_shifts(self, sensor, channel, shifts):
        with h5py.File(self.filename, 'a') as f:
            f.create_dataset('Channels/' + str(sensor) + '_' + str(channel) + '/shifts', data=shifts, dtype='int16')

    def check_key_names(self, sensors, channels, points, components):
        if isinstance(sensors, str) and sensors == 'all':
            sensors = self.sensors
        elif (not isinstance(sensors, list)) and (sensors not in self.sensors):
            raise Exception('Sensor name is incorrect:', sensors, 'self.sensors: ', self.sensors)
        elif isinstance(sensors, list) and (not set(sensors).issubset(self.sensors)):
            raise Exception('Sensors names is incorrect:', sensors, 'self.sensors: ', self.sensors)

        if isinstance(channels, str) and channels == 'all':
            channels = self.channels
        elif (not isinstance(channels, list)) and (channels not in self.channels):
            raise Exception('Channel name is incorrect:', channels, 'self.channels: ', self.channels)
        elif isinstance(channels, list) and (not set(channels).issubset(self.channels)):
            raise Exception('Channels names is incorrect:', channels, 'self.channels: ', self.channels)

        if isinstance(points, str) and points == 'all':
            points = self.logi
        if isinstance(points, np.ndarray):
            points = list(points)
        if not isinstance(points, list):
            points = [points]
        if not set(points).issubset(self.logi):
            raise Exception('Points name is incorrect:', points, 'self.logi: ', self.logi)

        if isinstance(components, str) and components == 'all':
            components = self.components
        if isinstance(components, str):
            components = [components]
        if not set(components).issubset(self.components):
            raise Exception('Сomponents name is incorrect:', components, 'self.components:', self.components)

        return sensors, channels, points, components

    def get_data(self, sensors, channels, points='all', components='all'):
        """ClassObject.get_data() - возвращает данные в том формате, в котором они находятся в файле,
        форма (N_points, N_components, L)
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            if components == self.components and points == self.logi:
                with h5py.File(self.filename, 'r') as f:
                    return f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:]
            elif set(components).issubset(self.components):
                if set(points).issubset(self.logi):
                    if components != self.components and points == self.logi:
                        with h5py.File(self.filename, 'r') as f:
                            L = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'].shape[-1]
                            dtype = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:].dtype
                            data = np.zeros((len(points), len(components), L), dtype=dtype)
                            for component in components:
                                data[:, components.index(component), :] = \
                                    f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:,
                                    self.components.index(component),
                                    :]
                            return data

                    elif components == self.components and points != self.logi:
                        with h5py.File(self.filename, 'r') as f:
                            L = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'].shape[-1]
                            dtype = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:].dtype
                            data = np.zeros((len(points), len(components), L), dtype=dtype)
                            for point in points:
                                data[points.index(point), :, :] = \
                                    f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][self.logi.index(point),
                                    :,
                                    :]
                            return data

                    elif components != self.components and points != self.logi:
                        with h5py.File(self.filename, 'r') as f:
                            L = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'].shape[-1]
                            dtype = f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:].dtype
                            data = np.zeros((len(points), len(components), L), dtype=dtype)
                            for point in points:
                                for component in components:
                                    data[points.index(point), components.index(component), :] = \
                                        f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][
                                        self.logi.index(point),
                                        self.components.index(component),
                                        :]
                            return data


                else:
                    raise Exception('Points name is incorrect')
            else:
                raise Exception('Сomponent name is incorrect')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_data(sensor, channel, points, components))
            return all_data

    def set_data(self, data, sensors, channels):
        """ClassObject.get_data() - изменяет данные в файле,
        количество points и components должно совпадать в записанных данных в файле и в передаваемых
        тип данных data_type так же должен совпадать

        :param data: данные списком [sen1_ch1, sen1_ch2,...] форма (N_points, N_components, L)
        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        """
        points = 'all'
        components = 'all'
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            with h5py.File(self.filename, 'a') as f:
                f['Channels/' + str(sensor) + '_' + str(channel) + '/data'][:] = data
        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            if not isinstance(data, list):
                data = [data]
            i = 0
            for sensor in sensors:
                for channel in channels:
                    self.set_data(data[i], sensor, channel)
                    i += 1

    def get_full_wave_time(self, sensors, channels, points='all', components='all', **kwargs):
        """ClassObject.get_full_wave_time() - возвращает полное временное представление сигнала,
        форма (N_points, N_components, L)
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        **kwargs: именованные аргументы - необязательные для некоторых типов сигнала параметры - тоже что и в __init__
        """
        self.load_attributes(kwargs)
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)
        if 'shifts' in kwargs:
            self.set_shifts(sensors, channels, kwargs['shifts'])

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            data = self.get_data(sensor, channel, points, components)
            if self.data_type == 'full_wave_time':
                return data
            elif self.data_type == 'window_shift_time':
                return self.restore_data(data, sensor, channel, points, components)
            elif self.data_type == 'full_wave_frequency':
                return self.freq_to_time(data, self.L, self.fmin, self.fmax)
            elif self.data_type == 'window_shift_frequency':
                data = self.freq_to_time(data, self.L_win, self.fmin, self.fmax)
                return self.restore_data(data, sensor, channel, points, components)
            else:
                raise Exception('Incorrect value of data_type, you must pass one of these: '
                                'full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_full_wave_time(sensor, channel, points, components))
            return all_data

    def get_window_shift_time(self, sensors, channels, points='all', components='all', **kwargs):
        """ClassObject.get_window_shift_time() - возвращает временное представление сигнала обрезанного по окнам
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        self.load_attributes(kwargs)
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)
        if 'shifts' in kwargs:
            self.set_shifts(sensors, channels, kwargs['shifts'])

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            data = self.get_data(sensor, channel, points, components)
            if 'shifts' in kwargs:
                self.set_shifts(sensor, channel, kwargs['shifts'])
            if self.data_type == 'full_wave_time':
                return self.cut_data(data, sensor, channel, points, components)
            elif self.data_type == 'window_shift_time':
                return data
            elif self.data_type == 'full_wave_frequency':
                data = self.freq_to_time(data, self.L, self.fmin, self.fmax)
                return self.cut_data(data, sensor, channel, points, components)
            elif self.data_type == 'window_shift_frequency':
                return self.freq_to_time(data, self.L_win, self.fmin, self.fmax)
            else:
                raise Exception('Incorrect value of data_type, you must pass one of these: '
                                'full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_window_shift_time(sensor, channel, points, components))
            return all_data

    def get_full_wave_frequency(self, sensors, channels, points='all', components='all', **kwargs):
        """ClassObject.get_full_wave_frequency() - возвращает частотное представление полного сигнала
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        self.load_attributes(kwargs)
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)
        if 'shifts' in kwargs:
            self.set_shifts(sensors, channels, kwargs['shifts'])

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            data = self.get_data(sensor, channel, points, components)
            if 'shifts' in kwargs:
                self.set_shifts(sensor, channel, kwargs['shifts'])
            if self.data_type == 'full_wave_time':
                return self.time_to_freq(data, self.L, self.fmin, self.fmax)
            elif self.data_type == 'window_shift_time':
                data = self.restore_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L, self.fmin, self.fmax)
            elif self.data_type == 'full_wave_frequency':
                return data
            elif self.data_type == 'window_shift_frequency':
                data = self.freq_to_time(data, self.L_win, self.fmin, self.fmax)
                data = self.restore_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L, self.fmin, self.fmax)
            else:
                raise Exception('Incorrect value of data_type, you must pass one of these: '
                                'full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_full_wave_frequency(sensor, channel, points, components))
            return all_data

    def get_window_shift_frequency(self, sensors, channels, points='all', components='all', **kwargs):
        """ClassObject.get_window_shift_frequency() - частотное представление сигнала, обрезанного по окнам
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        self.load_attributes(kwargs)
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)
        if 'shifts' in kwargs:
            self.set_shifts(sensors, channels, kwargs['shifts'])

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            data = self.get_data(sensor, channel, points, components)
            if 'shifts' in kwargs:
                self.set_shifts(sensor, channel, kwargs['shifts'])
            if self.data_type == 'full_wave_time':
                data = self.cut_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L_win, self.fmin, self.fmax)
            elif self.data_type == 'window_shift_time':
                return self.time_to_freq(data, self.L_win, self.fmin, self.fmax)
            elif self.data_type == 'full_wave_frequency':
                data = self.freq_to_time(data, self.L, self.fmin, self.fmax)
                data = self.cut_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L_win, self.fmin, self.fmax)
            elif self.data_type == 'window_shift_frequency':
                return data
            else:
                raise Exception('Incorrect value of data_type, you must pass one of these: '
                                'full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_window_shift_frequency(sensor, channel, points, components))
            return all_data

    def get_win_sh_freq_cut(self, freq_min, freq_max, sensors, channels, points='all', components='all', **kwargs):
        """ClassObject.get_win_sh_freq_cut() - частотное представление сигнала, обрезанного по окнам
        в новом частотном диапазоне (отличном от сохраненного в ClassObject)
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :params freq_min, freq_max: новый частотный диапазон
        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param points: необязательный параметр, по умолчанию 'all' - для всех точек визуализации, можно указать список
        конкретных точек, например [1, 2, 5, 7] и т.д.
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        self.load_attributes(kwargs)
        sensors, channels, points, components = self.check_key_names(sensors, channels, points, components)
        if 'shifts' in kwargs:
            self.set_shifts(sensors, channels, kwargs['shifts'])

        if (not isinstance(sensors, list)) and (not isinstance(channels, list)):
            sensor = sensors
            channel = channels
            data = self.get_data(sensor, channel, points, components)
            if 'shifts' in kwargs:
                self.set_shifts(sensor, channel, kwargs['shifts'])
            if self.data_type == 'full_wave_time':
                data = self.cut_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L_win, freq_min, freq_max)
            elif self.data_type == 'window_shift_time':
                return self.time_to_freq(data, self.L_win, freq_min, freq_max)
            elif self.data_type == 'full_wave_frequency':
                data = self.freq_to_time(data, self.L, self.fmin, self.fmax)
                data = self.cut_data(data, sensor, channel, points, components)
                return self.time_to_freq(data, self.L_win, freq_min, freq_max)
            elif self.data_type == 'window_shift_frequency':
                data = self.freq_to_time(data, self.L_win, self.fmin, self.fmax)
                return self.time_to_freq(data, self.L_win, freq_min, freq_max)
            else:
                raise Exception('Incorrect value of data_type, you must pass one of these: '
                                'full_wave_time, full_wave_frequency, window_shift_time, window_shift_frequency')

        else:
            if not isinstance(sensors, list):
                sensors = [sensors]
            if not isinstance(channels, list):
                channels = [channels]
            all_data = []
            for sensor in sensors:
                for channel in channels:
                    all_data.append(self.get_win_sh_freq_cut(freq_min, freq_max, sensor, channel, points, components))
            return all_data

    def cut_data(self, data, sensor, channel, points, components):
        """Внутренний метод класса, не используйте во внешних скриптах без необходимости
        """
        shifts = self.get_shifts(sensor, channel)
        data_cut = np.zeros((len(points), len(components), self.L_win), dtype=data.dtype)
        for point in points:
            for component in components:
                point_index = self.logi.index(point)
                component_index = self.components.index(component)
                window_start = int(shifts[point_index, component_index])
                window_end = int(shifts[point_index, component_index] + self.L_win)
                data_cut[points.index(point), components.index(component), :] = \
                    data[points.index(point), components.index(component), window_start:window_end]
        return data_cut

    def restore_data(self, data, sensor, channel, points, components):
        """Внутренний метод класса, не используйте во внешних скриптах без необходимости
        """
        if not hasattr(self, 'L_win'):
            raise Exception('You must pass L_win=L_win parameter')
        shifts = self.get_shifts(sensor, channel)
        restored_data = np.zeros((len(points), len(components), self.L), dtype=data.dtype)
        for point in points:
            for component in components:
                point_index = self.logi.index(point)
                component_index = self.components.index(component)
                window_start = int(shifts[point_index, 0])
                if window_start + self.L_win >= self.L:
                    window_start = int(self.L - self.L_win)
                window_end = window_start + int(self.L_win)
                restored_data[points.index(point), components.index(component), window_start:window_end] = \
                    data[points.index(point), components.index(component), :]
        return restored_data

    def time_to_freq(self, data, L, fmin, fmax):
        """Внутренний метод класса, не используйте во внешних скриптах без необходимости
        """
        # M=int(M);
        L = int(L)
        if L & 1:
            Nr = int(L + 1) / 2
            # Nf=Nr+M;
        else:
            Nr = int(L / 2) + 1
            # Nf=Nr+2*M;
        ii = np.arange(Nr)
        ff = ii * self.fd / L
        fmask = (fmin <= ff) & (ff < fmax)
        imask = ii[fmask]
        fbeg = imask[0]
        fend = imask[-1] + 1
        # print('imask: ', imask)
        # fbeg = floor(L * (self.fmin / self.fd + 0.5))
        # fend = floor(L * (self.fmax / self.fd + 0.5))
        # print('fbeg: ', fbeg)
        # print('fend: ', fend)
        return np.array(np.fft.rfft(data, n=L, axis=-1, norm="ortho"), dtype=np.complex64)[:, :, fmask]

    def freq_to_time(self, data, L, fmin, fmax):
        """Внутренний метод класса, не используйте во внешних скриптах без необходимости
        """
        # M=int(M);
        L = int(L)
        if L & 1:
            Nr = int(L + 1) / 2
            # Nf=Nr+M;
        else:
            Nr = int(L / 2) + 1
            # Nf=Nr+2*M;
        ii = np.arange(Nr)
        ff = ii * self.fd / L
        fmask = (fmin <= ff) & (ff < fmax)
        imask = ii[fmask]
        Lf = fmask.size
        # fbeg = floor(L * (self.fmin / self.fd + 0.5))
        # fend = floor(L * (self.fmax / self.fd + 0.5))
        full_freqs = np.zeros((data.shape[0], data.shape[1], L), dtype=np.complex64)
        full_freqs[:, :, imask] = data
        # full_freqs[:, :, -fend:-fbeg] = np.flip(data.real - data.imag * 1.j, axis=-1)
        return np.array(np.fft.irfft(full_freqs, n=L, axis=-1, norm="ortho").real)

    def add_channel(self, data, data_type, sensor, channels, field, shifts='No_shifts'):
        """Добавление данных в класс(файл) из массива данных
        obj.add_channel(data=data_to_add, data_type=obj.data_type, sensor='sen1', channels='X', field=field,
                        shifts=shifts_to_add)

        :param data: сигнал np.array , форма (N_points, N_components, L), либо список list [data1, data2, ...] - по
        количеству каналов
        :param data_type: тип сигнала
        :param sensor: имя сенсора загружаемых данных, str: 'sen1'
        :param channels: имена каналов загружаемых данных, список строк: ['Z', 'X', 'Y']
        :param field: список из Xcoord, Ycoord, Altitude - для сенсора, например [165535, 556568, 200]
        :param shifts: смещения оклон для обрезанного сигнала, форма: (N_points, N_components), либо список list
        [shifts1, shifts2, ...] - по количеству каналов
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(shifts, str) and not isinstance(shifts, list):
            shifts = [shifts]
        if data_type != self.data_type:
            raise Exception(f"data_type: {data_type} should be the same as data_type in object: {self.data_type}")
        if not isinstance(channels, list):
            channels = [channels]
        with h5py.File(self.filename, 'a') as f:
            for channel in channels:
                if self.is_sen_chan_exist(sensor, channel):
                    raise Exception(f"Sensor {sensor} with channel {channel} is alredy exist")
                if channel not in self.channels:
                    new_channels_names = self.channels
                    new_channels_names.append(channel)
                    self.channels = new_channels_names
                if sensor not in self.sensors:
                    new_sensors_names = self.sensors
                    new_sensors_names.append(sensor)
                    self.sensors = new_sensors_names

            channel_g = f.create_group('Channels/' + str(sensor) + '_' + str(channel))
            channel_g.create_dataset('data', data=data[channels.index(channel)],
                                     dtype=data[channels.index(channel)].dtype)
            if not isinstance(shifts, str):
                channel_g.create_dataset('shifts', data=shifts[channels.index(channel)], dtype='int16')
            field_hdf5 = f['field']
            field_hdf5.resize((field_hdf5.shape[0] + 1, field_hdf5.shape[1]))
            field_hdf5[-1, :] = np.concatenate(([self.sensors.index(sensor),
                                                 self.channels.index(channel)], field))

    def add_hdf5_file(self, filename):
        """Добавление данных в класс(файл) из hdf5 файла

        :param filename: путь к hdf5 файлу
        """
        new_data = BachataClass(filename)
        if new_data.data_type != self.data_type:
            raise Exception(f"data_type in file: {new_data.data_type} should be the same as "
                            f"data_type in object: {self.data_type}")
        for field_row in new_data.field:
            sensor = new_data.sensors[int(field_row[0])]
            channel = new_data.channels[int(field_row[1])]
            if self.is_sen_chan_exist(sensor, channel):
                print('!' * 100)
                print(f'Sensor {sensor} and channel {channel} is alredy in object')
                print('skipping this step')
                print('_' * 100)
                continue
            if channel not in self.channels:
                new_channels_names = self.channels
                new_channels_names.append(channel)
                self.channels = new_channels_names
            if sensor not in self.sensors:
                new_sensors_names = self.sensors
                new_sensors_names.append(sensor)
                self.sensors = new_sensors_names
            with h5py.File(self.filename, 'a') as f:
                field_hdf5 = f['field']
                field_hdf5.resize((field_hdf5.shape[0] + 1, field_hdf5.shape[1]))
                field_hdf5[-1, :] = np.concatenate(([self.sensors.index(sensor),
                                                     self.channels.index(channel)], field_row[2:]))
                channel_g = f.create_group('Channels/' + str(sensor) + '_' + str(channel))
                dtn = new_data.get_data(sensor, channel)
                channel_g.create_dataset('data', data=dtn, dtype=dtn.dtype)
                if new_data.is_shifts(sensor, channel):
                    channel_g.create_dataset('shifts', data=new_data.get_shifts(sensor, channel), dtype='int16')

    def coord_to_point(self, x, y, z):
        """
        Внутренний метод класса, возвращает точку визуализации по локальным или глобальным координатам
        """
        if x > 5000 or y > 5000:
            print(f'Detected global coordinates in get__by_coords function, switching to local')
            if self.is_xycenter:
                x = x - self.xcenter
                y = y - self.ycenter
            else:
                raise Exception("Can't switch to local - no xcenter ycenter coordinates in hdf5 file")
        if self.is_domain():
            x_idx = (np.abs(self.domainX - x)).argmin()
            y_idx = (np.abs(self.domainY - y)).argmin()
            z_idx = (np.abs(self.domainZ - z)).argmin()
            point = z_idx * self.domainX.shape[0] * self.domainY.shape[0] + x_idx * self.domainY.shape[0] + y_idx
            return point
        else:
            raise Exception(f'One of domainX, domainY, domainZ is not in hdf5 file')

    def get_data_by_coords(self, x, y, z, sensors, channels, components='all'):
        """ClassObject.get_data_by_coords() - возвращает данные в том формате, в котором они находятся в файле,
        форма (N_points, N_components, L)
        Вместо параметра point принимает локальные координаты из Domain
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param x: локальные координаты из DomainX
        :param y: локальные координаты из DomainY
        :param z: локальные координаты из DomainZ
        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        point = self.coord_to_point(x, y, z)
        return self.get_data(sensors, channels, point, components)

    def get_full_wave_time_by_coords(self, x, y, z, sensors, channels, components='all'):
        """ClassObject.get_full_wave_time_by_coords() - возвращает полное временное представление сигнала,
        форма (N_points, N_components, L)
        Вместо параметра point принимает локальные координаты из Domain
        Если sensors и/или channels - списки или 'all' - возвращает список list данных: [dataSen1Ch1, dataSen1Ch2, ...]

        :param x: локальные координаты из DomainX
        :param y: локальные координаты из DomainY
        :param z: локальные координаты из DomainZ
        :param sensors: имя сенсора, данные которого необходимо вернуть, либо список имен, либо 'all' для всех
        :param channels: имя канала, который необходимо вернуть, либо список имен, либо 'all' для всех
        :param components: необязательный параметр, по умолчанию 'all' - для всех компонент, можно указать список
        конкретных компонент, например ['xz', 'xy', 'xx', 'xyz'] и т.д.
        """
        point = self.coord_to_point(x, y, z)
        return self.get_full_wave_time(sensors, channels, point, components)

    def is_sen_chan_exist(self, sensor, channel):
        with h5py.File(self.filename, 'r') as f:
            return str(sensor) + '_' + str(channel) in f['Channels/'].keys()

    def get_field(self, sensor_name):
        """ возвращает поле field для выбранного сенсора по имени

        :param sensor_name: имя сенсора, строка str
        :return:
        """
        index = np.where(self.field[:, 0] == self.sensors.index(sensor_name))[0][0]
        return self.field[index, 2:]