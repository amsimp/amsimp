"""
AMSIMP Travis Test - Ensure AMSIMP is functioning correctly, if not, travis
build will fail.
"""

import amsimp

amsimp.Backend(3).pressure_thickness()
amsimp.Wind(3).geostrophic_wind()
amsimp.Water(3).precipitable_water()
