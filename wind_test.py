import amsimp
import metpy.calc as c
from metpy.units import units
import numpy as np

detail = amsimp.Wind(delta_latitude=5, delta_longitude=5)

u, v = detail.geostrophic_wind()
geostophic_vorticity = detail.geostophic_vorticity()

for i in range(len(u)):
    u_amsimp, v_amsimp = u[i], v[i]
    geostophic_amsimp = geostophic_vorticity[i]

    lat = detail.latitude_lines()
    lon = detail.longitude_lines()
    dx, dy = c.lat_lon_grid_deltas(longitude=lon, latitude=lat)
    height = detail.geopotential_height().value * units.m
    f = np.resize(
        detail.coriolis_parameter().value, (
            height.shape[0],
            height.shape[2],
            height.shape[1],
        )
    ) / units.s
    f = np.transpose(f, (0, 2, 1)) 

    u_metpy, v_metpy = c.geostrophic_wind(heights=height[i], f=f[i], dx=dx, dy=dy)
    geostophic_metpy = c.vorticity(u=u_metpy, v=v_metpy, dx=dx, dy=dy).magnitude * geostophic_amsimp.unit
    u_metpy, v_metpy = (u_metpy.magnitude * (detail.units.m / detail.units.s)), (v_metpy.magnitude * (detail.units.m / detail.units.s))

    print("Wind")
    diff_u, diff_v = np.abs(u_metpy - u_amsimp), np.abs(v_metpy - v_amsimp)
    u_mean, v_mean = np.mean(np.abs(u_metpy)), np.mean(np.abs(v_metpy))
    print(np.mean(diff_u), np.mean(diff_v))
    print(u_mean, v_mean)

    print("Geostophic Vorticity")
    diff = np.abs(geostophic_metpy - geostophic_amsimp)
    mean = np.mean(np.abs(geostophic_metpy))
    print(np.mean(diff))
    print(mean)

