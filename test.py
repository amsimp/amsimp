# Numerical Poisson Equation Solution using Finite Difference Method
import numpy as np
import amsimp
import sys
from datetime import datetime

nt = 1E4
date = datetime(2020, 6, 4, 12)
detail = amsimp.Wind(delta_latitude=3, delta_longitude=3, input_date=date)
detail.remove_all_files()
height_guess = detail.geopotential_height()
height_naive = height_guess
height_bound = height_guess
detail.remove_all_files()

detail = amsimp.Wind(delta_latitude=3, delta_longitude=3)

actual_height = detail.geopotential_height()
f = np.resize(
    detail.coriolis_parameter().value, (
        actual_height.shape[0],
        actual_height.shape[2],
        actual_height.shape[1],
    )
) / detail.units.s
f = np.transpose(f, (0, 2, 1))
geostophic_vorticity = (
    detail.geostophic_vorticity() * f
) / detail.gravitational_acceleration()
height = height_guess
ny = height.shape[1]
nx = height.shape[2]

lat = np.radians(detail.latitude_lines().value)
lat = np.resize(
    lat, (
        geostophic_vorticity.shape[0],
        geostophic_vorticity.shape[2],
        geostophic_vorticity.shape[1],
    )
)
dy = detail.a * np.gradient(lat, axis=2)
dy = np.transpose(dy, (0, 2, 1))

lon = np.radians(10)
dx_y = detail.a * np.cos(lat)
dx_y = np.resize(
    dx_y, (
        geostophic_vorticity.shape[0],
        geostophic_vorticity.shape[2],
        geostophic_vorticity.shape[1],
    )
)
dx = dx_y * lon
dx = np.transpose(dx, (0, 2, 1))

for it in range(int(nt)):
    height_guess = height.copy()

    height[:, 1:-1, 1:-1] = (
        (
            (
                height_guess[:, 1:-1, 2:] + height_guess[:, 1:-1, :-2]
            ) * dy[:, 1:-1, 1:-1]**2 + (
                height_guess[:, 2:, 1:-1] + height_guess[:, :-2, 1:-1]
            ) * dx[:, 1:-1, 1:-1]**2 - geostophic_vorticity[:, 1:-1, 1:-1] * dx[:, 1:-1, 1:-1]**2 * dy[:, 1:-1, 1:-1]**2
        ) / (2 * (dx[:, 1:-1, 1:-1]**2 + dy[:, 1:-1, 1:-1]**2))
    )

    height[:, 0, :] = height_bound[:, 0, :]
    height[:, ny-1, :] = height_bound[:, ny-1, :]
    height[:, :, 0] = height_bound[:, :, 0]
    height[:, :, nx-1] = height_bound[:, :, nx-1]

print(" ")
diff = actual_height - height
diff = np.abs(diff)
print(np.mean(diff))

diff = actual_height - height_naive
diff = np.abs(diff)
print(np.mean(diff))
