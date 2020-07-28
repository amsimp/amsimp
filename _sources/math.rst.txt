===========
Mathematics
===========

Not fully fleshed out, more details to be added.

Pressure Thickness
==================

Pressure Thickness is the measurement of the distance (in metres)
between any two constant pressure surfaces.

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/thickness_def.png
   :alt: Pressure Thickness Definition (provided by the NWS of the USA)
   :width: 95%
   :align: center

One of the most common thickness charts used in meteorology is the
1000-500 hPa thickness, and for the purposes of this project, it will be
the sole interest. This is the distance between the elevation of the
1,000 hPa and 500 hPa levels. Typically, the 1,000 hPa surface is used
to represent sea level but this is just a generalisation. On pressure
charts, the last digit (zero) of a thickness value is typically
truncated. So, a 1000-500 thickness value of 570 means the distance
between the two surfaces is 5,700 metres. The 1000-500 hPa thickness
value of 540 is traditionally used to determine rain versus snow. If
precipitation is predicted poleward of this 540-thickness line (if the
thickness value is less than 540), it is expected that it will be snow.
If precipitation is predicted on the equator side of this line (if the
thickness value is greater than 540), then it is expected that the
precipitation will be in a liquid form. The reason one is able to make
such an expectation is due to the fact that the 540-thickness line
closely follows the surface freezing temperature of 273
K :cite:`thickness`.

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/rainsnow_line.png
   :alt: Rain/Snow Line (provided by the NWS of the USA)
   :width: 95%
   :align: center

To determine the pressure thickness between two constant pressure
surfaces, the Hypsometric equation is utilised and as shown in equation
:eq:`hypsometric`.

The hypsometric equation relates an atmospheric pressure ratio to the
equivalent thickness of an atmospheric layer considering the layer mean
of virtual temperature, gravity, and occasionally wind. It is derived
from the hydrostatic equation and the ideal gas law.

.. math:: h = \Phi_2 - \Phi_1 = \frac{R \bar{T_v}}{g} \ln{\frac{p_1}{p_2}}
   :label: hypsometric 

Precipitable Water
==================

Precipitable Water is the total atmospheric water vapour contained in a
vertical column of unit cross-sectional area extending between any two
specified pressure levels.

Based on the definition, precipitable water can be described
mathematically as being:

.. math:: W = \int_{0}^{z} \rho_v dz
   :label: pwv_1

where :math:`\rho_v` is the density of water vapour, and where
:math:`\rho_v` is defined as:

.. math:: \rho_v = \frac{\texttt{mass of vapour}}{\texttt{unit volume}}

Following which, the hydrostatic equation can be applied to equation
:eq:`pwv_1` in order to replace :math:`dz` with :math:`dp`. The
reason for doing this is that atmospheric pressure is extremely easier
to measure, with devices such as weather balloons being readily
available.

.. math:: W = -\int_{p_1}^{p_2} \frac{\rho_v}{\rho g} dp
   :label: pwv_2

Where :math:`p_1` and :math:`p_2` are constant pressure surfaces, and
where :math:`p_1 > p_2`. Substituting in the definition of density,
:math:`\rho_v = \frac{m_v}{V}; \rho = \frac{m_{air}}{V}`, into equation
:eq:`pwv_2` results in:

.. math:: W = -\int_{p_1}^{p_2} \frac{1}{g} \frac{m_v V}{m_{air} V} dp

.. math:: \Rightarrow W = - \frac{1}{g} \int_{p_1}^{p_2} \frac{m_v}{m_{air}} dp

The integration term in this particular equation is the definition for
the specific humidity, with the units of measurement being
:math:`\frac{kg}{kg}`. The specific humidity can be approximated by the
mixing ratio, with an error typically around 4
% :cite:`pwv_def`.

Mixing Ratio is the ratio of the mass of a variable atmospheric
constituent to the mass of dry air.

.. math:: \therefore W = -\frac{1}{g} \int_{p_1}^{p_2} m dp
   :label: pwv_derive_fin

The units as given by the equation are :math:`\frac{kg}{m^2}`
(dimensionless), but, the preferred unit of measurement for rainfall is
:math:`mm`. The conversion between the two units of measurements is one
to one (:math:`1 \frac{kg}{m^2} = 1 mm`) In actual rainstorms,
particularly thunderstorms, amounts of rain very often exceed the total
precipitable water of the overlying atmosphere. This results from the
action of convergence that brings into the rainstorm the water vapour
from a surrounding area that is often quite large. Nevertheless, there
is general correlation between precipitation amounts in given storms and
the precipitable water of the air masses involved in those
storms :cite:`problems_with_pwv`.

For the purposes of numerically calculating the precipitable water for a
given column of air, equation :eq: pwv_derive_fin is
commonly rewritten as the following:

.. math:: W = -\frac{1}{\rho g} \int_{p_1}^{p_2} \frac{0.622 e}{p - e} dp
   :label: pwv

Vapour Pressure is the pressure exerted by a vapour when the vapour is
in equilibrium with the liquid or solid form, or both, of the same
substance. In meteorology, vapour pressure is used almost exclusively to
denote the partial pressure of water vapour in the atmosphere.

Considering vapor pressure is not one of the three prognostic variables
defined in section, it is therefore necessary to express vapor pressure
in terms of the already defined variables. This can be done through the
utilisation of temperature and relative humidity. First, one must determine
the saturated vapour pressure. This can be done using the following
equation :cite:`balton`:

.. math:: e_s = 6.112 \exp(\frac{17.67 T}{T + 243.5})

After which, the vapor pressure can be calculated as follows:

.. math:: e = \frac{e_{s} r}{100}
   :label: vapor_pressure_eq

In wet periods, the precipitable water is particularly close to the
saturated precipitable water. In situations when the precipitable water
is close to the saturated precipitable water, the precipitable water
changes very little over the day. Saturated precipitable water also
makes calculations a whole lot simpler, as specific humidity data is
rather difficult to come by.

In regards to the numerical calculation of the precipitable water, the
SciPy method, scipy.integrate.quad, is utilised in order to determine
the definite integral in equation :eq: `pwv`. This method
integrates the function using a technique from the Fortran library,
QUADPACK :cite:`scipy_integrate`.

.. _virtual_section:

Virtual Temperature
===================

The virtual temperature is the temperature at which dry air would have
the same density as the moist air, at a given pressure. In other words,
two air samples with the same virtual temperature have the same density,
regardless of their actual temperature or relative humidity. Because
water vapor is less dense than dry air and warm air is less dense than
cool air, the virtual temperature is always greater than or equal to the
actual temperature.

In relation to this project, the virtual temperature will be extremely
useful as it will allow for the calculation of the evolution of relative
humidity through the utilisation of the thermodynamic advection equation. 
This is due to the fact that virtual temperature can be expressed in terms of
vapor pressure, which can be converted into relative humidity. This can
be seen as follows:

.. math:: T_v = \frac{T}{1 - \frac{0.378 e}{p}}

The equation becomes the following, by substituting equation
:eq:`vapor_pressure_eq` in and rearranging for
relative humidity:

.. math:: r = \frac{100 (p T_v - p T)}{0.378 e_s T_v}

Primitive Equations
===================

To be added.
