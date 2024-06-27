import numpy as np
import matplotlib.pyplot as plt

import math

def calculation_velocities(Z, property_data):
    property_data_depths = np.unique(np.array([float(line.split()[0]) for line in property_data]))
    prop_data = property_data[np.abs(Z - property_data_depths).argmin()]
    vp, vs = list(map(lambda x: float(x), prop_data.split()[-2:]))
    return vp, vs


with open("property.txt", "r") as f:
    property_data = f.readlines()
f.close()

with open("sensors.txt", "r") as f:
    sensors = f.readlines()
f.close()

with open("415_inkl_point.txt", "r") as f:
    inkl = f.readlines()
f.close()

inkl_data = []
for i in range(1, len(inkl)):
    inkl_data.append(list(map(lambda x: float(x), inkl[i].split("\t"))))

inkl_data = np.array(inkl_data)
sensors_depths = []
for line in sensors[6:]:  # 415 c 6, 413 c 8
    line = line.split()
    sensors_depths.append([int(line[0]), int(line[2]) / 100])

sensors_depths = np.array(sensors_depths)
sensors_depths_unique = np.unique(sensors_depths[:, 1])

half_well = sensors_depths[:sensors_depths.shape[0] // 2, :]

# loop
for depth_ind in range(50, 1200, 10):
    # depth_ind = 600

    depth = half_well[depth_ind, 1]  # глубина по стволу

    #print(f'submodel central sensor name {half_well[depth_ind, 0]}, submodel central depth, m {depth}')

    vp, vs = calculation_velocities(depth, property_data)
    #print(f"vp {vp}, vs {vs}")

    x_sens, y_sens, z_sens = 0, 0, depth
    ind = np.abs(inkl_data[:, 0] - depth).argmin()

    # x0_new, y0_new, z0_new = x_sens - inkl_data[ind, 5], y_sens - inkl_data[ind, 6], z_sens - inkl_data[ind, 0]  # новый центр координат по стволу скважины
    #x0_new, y0_new, z0_new = x_sens - inkl_data[ind, 5], y_sens - inkl_data[ind, 6], z_sens - z_sens  # новый центр координат по стволу скважины

    # sens
    x_sens0, z_sens0, x_sens1, z_sens1 = inkl_data[ind, 5], half_well[depth_ind, 1], inkl_data[ind+1, 5], half_well[depth_ind + 1, 1]
    # TVD vector
    z_tvd0, z_tvd1 = inkl_data[ind, 1], inkl_data[ind + 1, 1]

    sens_vect = np.sqrt((x_sens1 - x_sens0) ** 2 + (z_sens1 - z_sens0) ** 2)
    tvd_vect = np.sqrt((z_tvd1 - z_tvd0) ** 2)

    scalar = (z_sens1 - z_sens0)*(z_tvd1 - z_tvd0)

    rad = math.acos(scalar / (sens_vect * tvd_vect))
    print('angle', (math.acos(scalar / (sens_vect * tvd_vect))*180)/np.pi)

    x_sens_new = (x_sens1 - x_sens0)*math.cos(rad)-(z_sens1 - z_sens0)*math.sin(rad)
    z_sens_new = -(x_sens1 - x_sens0)*math.sin(rad)-(z_sens1 - z_sens0)*math.cos(rad)
    #
    # sens_vect = np.sqrt((x_sens_new - x_sens0) ** 2 + (z_sens_new - x_sens0) ** 2)
    #
    # scalar = z_sens0 * z_tvd0 + z_sens_new * z_tvd1
    # print('angle', (math.acos(scalar / (sens_vect * tvd_vect))*180)/np.pi)

pass
