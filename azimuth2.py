import numpy as np
import matplotlib.pyplot as plt

import green2

import math


def calculation_velocities(Z, property_data):
    property_data_depths = np.unique(np.array([float(line.split()[0]) for line in property_data]))
    prop_data = property_data[np.abs(Z - property_data_depths).argmin()]
    vp, vs = list(map(lambda x: float(x), prop_data.split()[-2:]))
    return vp, vs


def calc_r(x, y):
    return np.sqrt(x ** 2 + y ** 2)


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

#plt.plot(sensors_depths[:sensors_depths.shape[0] // 2, 1])
plt.plot(inkl_data[:, 1])
plt.plot(inkl_data[:, 0])
plt.gca().invert_yaxis()
plt.show()

# loop
for depth_ind in range(1150, 1200, 10):
    # depth_ind = 600

    depth = half_well[depth_ind, 1]  # глубина по стволу

    # print(f'submodel central sensor name {half_well[depth_ind, 0]}, submodel central depth, m {depth}')

    vp, vs = calculation_velocities(depth, property_data)
    # print(f"vp {vp}, vs {vs}")

    ind = np.abs(inkl_data[:, 0] - depth).argmin()

    # sens
    x_sens0, y_sens0, z_sens0, x_sens1, y_sens1, z_sens1 = inkl_data[ind, 5], inkl_data[ind, 6], half_well[
        depth_ind, 1], inkl_data[ind + 1, 5], inkl_data[ind + 1, 6], half_well[depth_ind + 1, 1]
    # TVD vector
    z_tvd0, z_tvd1 = inkl_data[ind, 1], inkl_data[ind + 1, 1]

    sens_vect = np.sqrt((x_sens1 - x_sens0) ** 2 + (y_sens1 - y_sens0) ** 2 + (z_sens1 - z_sens0) ** 2)
    tvd_vect = np.sqrt((z_tvd1 - z_tvd0) ** 2)

    scalar = (z_sens1 - z_sens0) * (z_tvd1 - z_tvd0)

    rad = math.acos(scalar / (sens_vect * tvd_vect))
    print('angle', (math.acos(scalar / (sens_vect * tvd_vect)) * 180) / np.pi)

    R = calc_r((x_sens1 - x_sens0), (y_sens1 - y_sens0))

    R_center_new = R * math.cos(rad) + (z_sens1 - z_sens0) * math.sin(rad)
    z_center_new = R * math.sin(rad) - (z_sens1 - z_sens0) * math.cos(rad)

    tmod = 3
    fd = 3000
    Q = 21
    Nd = 1
    radius = 10
    L = int(fd * tmod)

    sensors_coords = []
    for i in range(1000, 1100, 10):
        depth_sens = half_well[i, 1]
        ind_sens = np.abs(inkl_data[:, 0] - depth_sens).argmin()
        x_vect, y_vect, z_vect = (inkl_data[ind, 5] - inkl_data[ind_sens, 5]), (
                inkl_data[ind, 6] - inkl_data[ind_sens, 6]), (half_well[depth_ind, 1] - half_well[i, 1])
        R_sens = calc_r(x_vect, y_vect)

        R_sens_new = R_sens * math.cos(rad) + z_vect * math.sin(rad)
        z_sens_new = R_sens * math.sin(rad) - z_vect * math.cos(rad)

        sensors_coords.append([R_sens_new, y_vect, z_sens_new])

    sensors_coords = np.array(sensors_coords)

    sources_coords = np.array(
        [[R_center_new, y_sens1 - y_sens0, z_center_new - radius],
         [R_center_new, y_sens1 - y_sens0, z_center_new + radius]])

    Gtp_total = np.zeros((sources_coords.shape[0], sensors_coords.shape[0], 3, 6, L))
    Gts_total = np.zeros((sources_coords.shape[0], sensors_coords.shape[0], 3, 6, L))
    for num_sr in range(sources_coords.shape[0]):
        xx = sensors_coords - sources_coords[num_sr, :]
        rs = green2.response_t(xx, T=tmod, fs=fd)
        rs.make_green(Q=Q, vp=vp, vs=vs, fp=1e7, a3=1e-5, naive=False)

        Gtp = - rs.Gtp(nD=Nd)
        Gts = - rs.Gts(nD=Nd)

        Gtp_total[num_sr, :, :, :] = Gtp
        Gts_total[num_sr, :, :, :] = Gts

    pass
