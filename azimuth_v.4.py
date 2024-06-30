import numpy as np
import matplotlib.pyplot as plt

import green2

import math


def calculation_velocities(Z1, Z2, property_data):
    property_data_depths = np.unique(np.array([float(line.split()[1]) for line in property_data]))

    prop_data = property_data[np.abs(Z1 - property_data_depths).argmin():np.abs(Z2 - property_data_depths).argmin() + 1]

    Vp, Vs = [], []
    for i in range(len(prop_data)):
        vp, vs = list(map(lambda x: float(x), prop_data[i].split()[-2:]))
        Vp.append(vp)
        Vs.append(vs)

    Vp, Vs = np.mean(Vp), np.mean(Vs)

    return Vp, Vs


def calc_r(x, y):
    return np.sqrt(x ** 2 + y ** 2)


with open("../../Downloads/azimuth/property.txt", "r") as f:
    property_data = f.readlines()
f.close()

with open("../../Downloads/azimuth/sensors.txt", "r") as f:
    sensors = f.readlines()
f.close()

with open("../../Downloads/azimuth/415_inkl_point.txt", "r") as f:
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

old_well = []
for i in range(inkl_data.shape[0]):
    R = calc_r(inkl_data[i, 3], inkl_data[i, 4])
    TVD = inkl_data[i, 1]

    old_well.append([R, TVD])

old_well = np.array(old_well)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(old_well[:, 0], old_well[:, 1], label='old')
ax.set_xlabel('R')
ax.set_ylabel('TVD')
ax.set_title('Вид сбоку')
plt.gca().invert_yaxis()
ax.set_aspect(1)
plt.legend()
plt.show()
# ----------------ROTATION---------------------------------
depth_ind = 766
depth = half_well[depth_ind, 1]

ind = np.abs(inkl_data[:, 0] - depth).argmin()

R_sens0 = calc_r(inkl_data[ind, 3], inkl_data[ind, 4])
R_sens1 = calc_r(inkl_data[ind + 1, 3], inkl_data[ind + 1, 4])
z_sens0 = inkl_data[ind, 1]
z_sens1 = inkl_data[ind + 1, 1]

z_tvd0, z_tvd1 = inkl_data[ind, 1], inkl_data[ind + 1, 1]

sens_vect = np.sqrt((R_sens1 - R_sens0) ** 2 + (z_sens1 - z_sens0) ** 2)
tvd_vect = np.sqrt((z_tvd1 - z_tvd0) ** 2)

scalar = (z_sens1 - z_sens0) * (z_tvd1 - z_tvd0)

rad = - (math.acos(scalar / (sens_vect * tvd_vect)) + np.pi / 2)
print('angle', (math.acos(scalar / (sens_vect * tvd_vect)) * 180) / np.pi)

print("R_center_new", R_sens0, "z_center_new", z_sens0)

new_well = []
for i in range(inkl_data.shape[0]):
    R = calc_r(inkl_data[i, 3], inkl_data[i, 4])
    TVD = inkl_data[i, 1]

    R_coord = R - R_sens0
    TVD_coord = TVD - z_sens0

    long_coord_new = -(R_coord * math.cos(rad) + TVD_coord * math.sin(rad))
    perp_coord_new = R_coord * math.sin(rad) - TVD_coord * math.cos(rad)

    new_well.append([long_coord_new, perp_coord_new])

new_well = np.array(new_well)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(new_well[:, 0], new_well[:, 1], label='new')
ax.set_xlabel('long_coord')
ax.set_ylabel('perp_coord')
ax.set_title('Вид сбоку')
plt.gca().invert_yaxis()
ax.set_aspect(1)
plt.legend()
plt.show()
# ----------------------------------------------------------
radius = 100
tmod = 3
fd = 500
Q = 50
Nd = 1
L = int(fd * tmod)

# ------------velocity-------------------
# min_depth, max_depth = 0, np.max(np.abs(new_well[:, 1]))
# vp, vs = calculation_velocities(min_depth, max_depth, property_data)
vp, vs = 3000, 1500
# ---------------------------------------

sens_coords_new = []
for i in range(sensors_depths.shape[0]):
    depth = sensors_depths[i, 1]
    ind = np.abs(inkl_data[:, 0] - depth).argmin()
    long_sens, perp_sens = new_well[ind, 0], new_well[ind, 1]
    sens_coords_new.append([long_sens, 0, perp_sens])

sens_coords_new = np.array(sens_coords_new)[:200, :]

sources_coords = np.array([[R_sens0, 0, z_sens0 - radius], [R_sens0, 0, z_sens0 + radius]])

Gtp_total = np.zeros((sources_coords.shape[0], sens_coords_new.shape[0], 3, 6, L))
# Gts_total = np.zeros((sources_coords.shape[0], sens_coords_new.shape[0], 3, 6, L))
for num_sr in range(sources_coords.shape[0]):
    xx = sens_coords_new - sources_coords[num_sr, :]
    rs = green2.response_t(xx, T=tmod, fs=fd)
    rs.make_green(Q=Q, vp=vp, vs=vs, fp=1e7, a3=1e-5, naive=False)

    Gtp = - rs.Gtp(nD=Nd)
    # Gts = - rs.Gts(nD=Nd)

    Gtp_total[num_sr, :, :, :] = Gtp
    # Gts_total[num_sr, :, :, :] = Gts

pass
