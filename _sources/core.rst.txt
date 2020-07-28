.. role:: raw-latex(raw)
   :format: latex
..

.. _fdm_section:

==============
Dynamical Core
==============

Finite Difference Method
========================

Introduction
------------

The use of numerical models for the simulation of dynamics within the
atmosphere typically involves the solution of a set of partial
differential equations. These equations generally describe three
processes: advection, adjustment and diffusion.

Advection is the transport of a substance by bulk motion.

Adjustment is how the mass and wind fields adjust to one another.

Diffusion is the movement of a liquid, or gas from an area of high
concentration to an area of low concentration.

Most meteorological problems involving partial differential equations
generally fall into three distinct categories: initial value problems,
boundary value problems, and eigenvalue problems. The meteorological
problems associated with this project are solely initial value
problems :cite:`numerical_methods`. An initial value problem
is a situation where you want to predict the future state of a given
system, given the necessary initial conditions. Unfortunately, the
equations describing the evolution of the atmosphere do not have exact
analytical solutions.

An analytical function is the most precise way of representing a
physical field as it gives us the value of this field at any point in
space, and at any instant in time.

When an analytical solution does not exist, an approximate numerical
solution is found using a specified computational
technique :cite:`numerical_methods`. For the purposes of this
project, the finite difference method will be utilised. It must be
noted, however, that there are several other methods to acquire an
approximate numerical solution, with the finite element method and the
spectral method being just a few.

The idea of the finite difference method is to approximate the
derivatives in the partial differential equations with differences
between adjacent points in space and in time. The advantages of this
being that the problem becomes an algebraic one, and that a continuous
problem becomes a discrete one.

Derivation of the Finite Difference Method
------------------------------------------

In order to derive the finite difference method, it is necessary to look
at the Taylor series expansion of the function.

The Taylor series of a function is the limit of that function’s Taylor
polynomials as the degree increases.

The Taylor series expansion of :math:`f(x)` can be represented as the
following:

.. math:: f(x + \Delta x) = f(x) + \frac{d f}{d x}(x) \frac{\Delta x}{1!} + \frac{d^2 f}{d x^2}(x) \frac{\Delta x^2}{2!} + ... + \frac{d^n f}{d x^n}(x) \frac{\Delta x^n}{n!} + ...

.. math:: f(x - \Delta x) = f(x) + \frac{d f}{d x}(x) \frac{-\Delta x}{1!} + \frac{d^2 f}{d x^2}(x) \frac{(-\Delta x)^2}{2!} + ... + \frac{d^n f}{d x^n}(x) \frac{(-\Delta x)^n}{n!} + ...

The higher order terms, which will be represented as
:math:`\mathcal{O}(\Delta x)` from this point on, become less important
as :math:`\Delta x` approaches zero.

.. math:: \Rightarrow f(x + \Delta x) = f(x) + \frac{d f}{d x}(x) \frac{\Delta x}{1!} + \frac{d^2 f}{d x^2}(x) \frac{\Delta x^2}{2!} + [\mathcal{O}(\Delta x^3)] 
   :label: fds

.. math:: \Rightarrow f(x - \Delta x) = f(x) - \frac{d f}{d x}(x) \frac{\Delta x}{1!} + \frac{d^2 f}{d x^2}(x) \frac{\Delta x^2}{2!} + [\mathcal{O}(\Delta x^3)]
   :label: bds

These higher order terms are neglected, and the following approximation
for the derivative of :math:`f(x)` is found:

.. math:: \Rightarrow \frac{d f}{d x}(x) = \frac{f(x + \Delta x) - f(x)}{\Delta x} + \mathcal{O}(\Delta x)

.. math:: \Rightarrow \frac{d f}{d x}(x) = \frac{f(x - \Delta x) - f(x)}{\Delta x} + \mathcal{O}(\Delta x)

These are called the forward and backward difference schemes
respectively, and by keeping only the leading order terms, an error of
order :math:`\mathcal{O}(\Delta x)` is occurred. It is possible to
obtain a better approximation by subtracting :eq:`bds` from
:eq:`fds`, which yields equation :eq:`eq_fds_bds_subtract`.

.. math:: \frac{d f}{d x}(x) = \frac{f(x + \Delta x) - f(x - \Delta x)}{\Delta x} + \mathcal{O}(\Delta x^2)
   :label: eq_fds_bds_subtract

This particular approximation is called the central difference scheme,
and has an error of order :math:`\mathcal{O}(\Delta x^2)`. Therefore,
this scheme is more accurate than the previously mentioned forward and
backward difference schemes. It is possible to take more and more terms
from the Taylor series expansion, however, there is an inherent trade
off between accuracy and computational efficiency.

In relation to the atmosphere, what this method does is divide the
atmosphere into several discrete horizontal layers, and each layer is
divided up into grid cells. Following which, each equation is evaluated
at the centre of the cell. Similarly, the time interval under
consideration is sliced into a number of discrete time steps. The size
of the grid step :math:`\Delta x` and time step :math:`\Delta t`
determines the accuracy of the scheme, with accuracy increasing as
:math:`\Delta x` and :math:`\Delta t` approach zero. On a synoptic
scale, :math:`\Delta x` is generally equal to 500 km. For higher
resolutions, the grid-size is smaller, which corresponds to a greater
computational burden. As such, there is a trade off between accuracy and
computational performance. For Eulerian schemes, the typical time step
is 2 minutes. As such, since the software will use an Eulerian scheme,
the time step will be 2 minutes :cite:`leapfrog_slides_one`.

.. _ftcs_section:

FTCS Scheme
-----------

Given the information mentioned in the previous section, the most
obvious scheme to approximate a differential equation, which will be
used to predict the future state of the atmosphere, would be to combine
the central difference scheme for space and the forward difference
scheme for time (FTCS). This scheme would allow us access to the
increased accuracy of the central difference scheme, while maintaining
two time variable unknowns. If only it was that simple! Let’s take the
example of the 1-D linear advection equation for temperature. This
equation is represented as the following:

.. math:: \frac{\partial T}{\partial t} + u \frac{\partial T}{\partial x} = 0
   :label: 1d_temp_eq

Using the FTCS scheme mentioned above, this equation can be approximated
as:

.. math:: \frac{T^{n+1}_{i} - T^{n}_{i}}{\Delta t} + u \frac{T^{n}_{i+1} - T^{n}_{i-1}}{2 \Delta x} = 0

It can be shown, by using Fourier Series, that:

.. math:: |\lambda_j|^2 = 1 + \alpha^2(\sin{j \Delta x}^2)

Therefore, :math:`|\lambda_j|^2 \geq 1`, and so the scheme is said to be
absolutely unstable. What it means for a scheme to be unstable is that
if there is a slight change in the initial value, the result of the
computation will change dramatically. The stability of a scheme is
important in meteorological problems because if slight deviations from
the mathematical model caused by unavoidable errors in measurement do
not have a correspondingly slight effect on the approximate numerical
solution, the mathematical equations describing the problem will not
accurately predict the future outcome :cite:`ftcs_leapfrog`.
For a more detailed technical explanation of the stability of this
scheme and the leapfrog scheme, please see the following article:
https://www.ecmwf.int/sites/default/files/elibrary/2002/16948-numerical-methods.pdf.

.. _leapfrog:

Leapfrog Scheme
---------------

This scheme is probably the most common scheme used for meteorological
problems. The "leapfrog" refers to the centred time difference which is
used in conjunction with centred space differences.

Taking the 1-D linear advection equation for temperature seen in
equation :eq:`1d_temp_eq`, applying this scheme results
in:

.. math:: \frac{T^{n+1}_{i} - T^{n-1}_{i}}{2 \Delta t} + u \frac{T^{n}_{i+1} - T^{n}_{i-1}}{2 \Delta x} = 0

It can be shown that this scheme is stable using a similar technique
previously mentioned. This equation can then be
rearranged for the forecast value
:math:`T^{n+1}_{i}`\  :cite:`ftcs_leapfrog`:

.. math:: T^{n+1}_{i} = T^{n-1}_{i} - u \frac{2 \Delta t}{2 \Delta x}(T^{n}_{i+1} - T^{n}_{i-1})

For the physical equation, a single initial condition :math:`T^{0}` is
sufficient to determine the solution. One problem with the leapfrog
scheme is that two values of :math:`T` are required to start the
computation. In addition to the physical initial condition
:math:`T^{0}`, a computational initial condition :math:`T^{1}` is
required. This cannot be obtained using the leapfrog scheme, so a
non-centred step is used to provide the value at :math:`t = \Delta t`.
From which point on, the leapfrog scheme is used, however, the errors of
the first step will persist. This method, however, still retains an
error of order :math:`\mathcal{O}(\Delta t^2)`. If you also use half of
the time step for the forward time step, followed by leapfrog time
steps; this will reduce the error introduced in the first
step :cite:`leapfrog_slides_two`. This will be the method
utilised within the software.

Nonlinear Instability
---------------------

A major problem which occurs while dealing with nonlinear partial
differential equations is nonlinear instability. This is a problem where
there is a nonlinear interaction between atmospheric
waves :cite:`nonlinear_instability`.

An atmospheric wave is a periodic disturbance in the fields of
atmospheric variables (like geopotential height, temperature, or wind
velocity) which may either propagate (travelling wave) or not (standing
wave).

If one of the waves involved in this nonlinear interaction have a
wavelength less than :math:`4 \Delta x` something called aliasing causes
a channelling of energy towards the small wavelengths. The continuous
feedback of energy leads to a catastrophic rise in the kinetic energy of
wavelengths between :math:`2 \Delta x` and :math:`4 \Delta x`. Within
the software, a smoothing operator, which reduces the amplitude of the
short waves while having little effect on the meteorologically important
waves, is utilised :cite:`nonlinear_instability`.

Another problem to mention before moving on is that for nonlinear
equations, the leapfrog scheme has a tendency to increase the amplitude
of the computational mode with time This can separate the space
dependence between the even and odd time steps. This problem can be
rectified by applying a Robert-Asselin Time Filter. After
:math:`T^{n+1}` is obtained a slight time smoothing is applied to
:math:`T^{n}`, where :math:`\gamma` is on the order of
0.1 :cite:`leapfrog_slides_two`:

.. math:: T^{n} = T^{n} + \gamma(T^{n+1} - 2 T^{n} + T^{n-1})

Recurrent Neural Network
========================

.. _introduction-2:

Introduction
------------

Weather forecasting has traditionally been done by physical models of
the atmosphere, which are unstable to perturbations, and thus are
inaccurate for large periods of time :cite:`why_rnn`. Since
machine learning techniques are more robust to perturbations, it would
be logical to combine a neural network with a physical model. Weather
forecasting is a sequential data problem, therefore, a recurrent neural
network is the most suitable option for this task.

A recurrent neural network is a class of artificial neural networks
where connections between nodes form a directed graph along a temporal
sequence.

Before, we delve into the specific example of using a recurrent neural
network to predict the future state of the atmosphere, it is necessary
to review what a recurrent neural network is. Recurrent Neural Networks
(RNNs) are neural networks that are used in situations where data is
presented in a sequence. For example, let’s say you want to predict the
future position of a fast-moving ball. Without information on the
previous position of the ball, it is only possible to make an inaccurate
guess. If you had, however, a large number of snapshots of the previous
position, you are then able to predict the future position of the ball
with some certainty. RNNs excel at modelling sequential data such as
these. This is due to sequential memory.

In order to intuitively understand sequential memory, the prime example
would be the alphabet. While it is easy to say the alphabet from A-Z, it
is much harder to go from Z-A. There is a logical reason why this is
difficult. As a child, you learn the alphabet in a sequence. Sequential
memory is a mechanism that makes it easier for your brain to recognise
sequence patterns.

In a traditional neural network, there is a input layer, hidden layer,
and a output layer. In a recurrent neural network, a loop is added that
can be added to pass information forward as seen in the diagram below
(provided by Towards Data Science) :cite:`intro_rnn`:

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/rnn.png
   :alt: Visualisation of a Recurrent Neural Network
   :width: 20%
   :align: center

The information that is forwarded is the hidden layer, which is a
representation of previous inputs. How this works in practise is that
you initialise your network layers and the hidden the initial hidden
state. The shape and dimension of the hidden state will be dependent on
the shape and dimension of your recurrent neural network. Then you loop
through your inputs, pass the relevant parameter and hidden state into
the RNN. The RNN returns the output and a modified hidden state. Last
you pass the output to the output layer, and it returns a prediction.

There is, however, a major problem known as short-term memory.
Short-term memory is caused by something known as the vanishing gradient
problem, which is also prevalent in other neural network architectures.
As the RNN processes more steps, it has troubles retaining information
from previous steps. Short-Term memory and the vanishing gradient is due
to the nature of back-propagation. This can be comprehended through
understanding how a neural network is
trained :cite:`intro_rnn`.

Back-propagation is an algorithm used to train and optimise neural
networks.

To train a recurrent neural network, you use an application of
back-propagation called back-propagation through time. Training a neural
network has three major steps. First, the relevant data vector is
normalised between 0 and 1, the vector is feed into the RNN, and it goes
through an activation function. The activation function utilised in the
software is the rectified linear activation
function :cite:`lstm_rnn`.

The rectified linear activation function is a piece-wise linear function
that will output the input directly if is positive, otherwise, it will
output zero.

The function is linear for values greater than zero, meaning it has a
lot of the desirable properties of a linear activation function when
training a neural network using back-propagation. Yet, it is a nonlinear
function as negative values are always output as zero. As a result, the
rectified function is linear for half of the input domain and nonlinear
for the other half, it is referred to as a piece-wise linear
function :cite:`relu`. This nonlinear element is extremely
important if the system has a nonlinear component, for example in
predicting the evolution of the future state of the atmosphere.

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/relu.png
   :alt: Sketch of the Rectified Linear Activation Function
   :width: 95%
   :align: center

Second, it outputs the results. Third, it compares the prediction to the
ground truth using a loss function.

A loss function outputs an error value which is an estimate of how
poorly the network is performing.

The lost function that will be utilised in the software will be the
function for mean squared error. The reason for choosing this particular
function is that it heavily penalises large errors, as it squares the
difference between the predicted and actual value. A large error in a
weather forecast is highly undesirable, hence, the use of this function.
The function is represented below:

.. math:: MSE = \frac{1}{n}\sum_{i=1}^n(Y_i-\hat{Y_i})^2

If a vector of :math:`n` predictions is generated from a sample of
:math:`n` data points on all variables, and :math:`Y` is the vector of
observed values of the variable being predicted, with :math:`\hat{Y_i}`
being the predicted values.

Mean squared error is the average squared difference between the
estimated values and the actual value.

Returning to the training of the RNN, it uses that error value from the
loss function. to do back propagation which calculates the gradients for
each time step in the network. The gradient is the value used to adjust
the networks internal weights, allowing the network to learn. The bigger
the gradient, the bigger the adjustments and vice versa. Here is where
the problem lies. When doing back propagation, the gradient of the
current time step is calculated with respect to the effects of the
gradients, in the time step before it. So if the adjustments to the time
step before it is small, then adjustments to the current time step will
be even smaller. The gradient values will exponentially shrink as it
propagates through each time step. That causes gradients to
exponentially shrink as it back propagates down. The earlier layers fail
to do any learning as the internal weights are barely being adjusted due
to extremely small gradients.

Because of vanishing gradients, the RNN doesn’t learn the long-range
dependencies across time steps. So not being able to learn on earlier
time steps causes the network to have a short-term memory. In order to
combat this, a long short-term memory is
used :cite:`intro_rnn`.

LSTM
----

LSTM’s were created as a solution to the short-term memory problem. They
have internal mechanisms called gates that can regulate the flow of
information. These gates can learn which data in a sequence is important
to keep or throw away. By doing that, it can pass relevant information
down the long chain of sequences to make predictions. For example, if
you were interested in buying a particular, you might read a review in
order to determine if the purchase of the product is a good decision.
When you read a review, your brain subconsciously only remembers
important keywords. You pick up words like “amazing", “superb", or
“awful", you don’t remember words such as "the", "as", or "because".
This is what an LSTM does, it learns to keep only the relevant
information to make predictions.

An LSTM has a similar control flow as a recurrent neural network. It
processes data passing on information as it propagates forward. The
differences are the operations within the LSTM’s cells. The core concept
of LSTM’s are the cell state, and it’s various gates. The cell state is
the method by which information is transferred down the sequence chain.
The cell state, in theory, can carry relevant information throughout the
processing of the sequence. So even information from the earlier time
steps can make it’s way to later time steps, reducing the effects of
short-term memory. As the cell state goes on its journey, information
get’s added or removed to the cell state via
gates :cite:`lstm_rnn`.

A gate is an electric circuit with an output which depends on the
combination of several inputs.

Gates contain the sigmoid activation function. The sigmoid activation
function squishes values between 0 and 1. That is helpful to update or
forget data because any number getting multiplied by 0 is 0, causing
values to disappears or be “forgotten". Any number multiplied by 1 is
the same value therefore that value stay’s the same or is “kept".

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/sigmoid.png
   :alt: Sketch of the Sigmoid Activation Function
   :width: 95%
   :align: center

There are three types of gates utilised within a neural network: a
forget gate, an input gate, and an output gate. A forget gate decides
what information should be thrown away or kept. Information from the
previous hidden state and information from the current input is passed
through the sigmoid function. An input gate is where the previous hidden
state and current input into a sigmoid function. The output gate decides
what the next hidden state should be. The hidden state is also used for
predictions. First, we pass the previous hidden state and the current
input into a sigmoid function. Then we pass the newly modified cell
state to the rectified linear activation function. We multiply the
rectified linear activation function output with the sigmoid output to
decide what information the hidden state should carry. The output is the
hidden state. The new cell state and the new hidden is then carried over
to the next time step :cite:`lstm_rnn`.

.. _implement_rnn:

Implementation
--------------

The data set for the initial conditions consists of three features:
geopotential height, air temperature, and relative humidity. For the
purposes of this specific project, the RNN will solely be trained on air
temperature and relative humidity. Unfortunately, due to the COVID-19,
there was a time constraint on the developed of the RNN, which resulted
in the inability to also train the RNN on geopotential height. This is
due to the lack of computational resources at my disposable. The data
set in question is updated every six hours by the National Oceanic and
Atmospheric Administration. This means for a single day, there will be
four observations. The goal for this project will be to, first predict
the relevant atmospheric parameter in seven days time given the last
thirty days of data and combine this RNN prediction with the physical
model prediction in an attempt to make a more accurate prediction
overall. In order to make such predictions, it is necessary to create a
window of the last 120 (:math:`30 \times 4`) observations to train the
model :cite:`time_series`.

At the start, a seed is set in order to ensure reproducibility. As
mentioned previously, it is important to scale features before training
a neural network. Normalisation is a common way of doing this scaling by
subtracting the mean and dividing by the standard deviation of each
feature. In order for the most optimal performance, the method
“MinMaxScaler" from the library, scikit-learn, is utilised within the
software :cite:`scikit-learn`. An LSTM requires a
1-dimensional sequence, however, the atmosphere is a 3-dimensional
system. Hence, it is necessary to flatten the 3-dimensional vector that
represents the state of the atmosphere. This is done in order to avoid
the need of repeatably running the RNN. Batches are then created to
split the data into manageable sequences. The diagram on the following
page shows how the data is represented after flattening the data and
batching it (provided by Tensorflow) :cite:`time_series`.

.. figure:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Images/data_rnn.png
   :alt: Visualisation of how the data is represented after flattening and batching.
   :width: 95%
   :align: center

Following this process, the data is feed into the RNN. The LSTM model is
built using Keras in TensorFlow, which is an free and open-source
software library for machine learning. It was developed by the Google
Brain Team :cite:`tensorflow`. It is apparent that a
multi-step model is needed as the model needs to learn to predict a
range of future values. The source code for the LSTM model developed for
the software is shown below:

.. code:: python

   # Prepossessed historical data, which has been flatten and batched.
   x_data, y_data = prepossessing_function(input_data)
   # Prepossessed initial conditions, which has been flatten and batched.
   initial_conditions = prepossessing_function(input_initialconditions)

   # The network is shown data from the last 15 days.
   past_history = 15 * 4

   # The network predicts the next 7 days worth of steps.
   future_target = 7 * 4

   # Create, and train models.
   # Optimiser.
   opt = Adam(lr=1e-6, decay=1e-10, clipvalue=0.6)
   # Create model.
   model = Sequential()
   model.add(
       LSTM(
           400, activation='relu', input_shape=(past_history, features)
       )
   )
   model.add(RepeatVector(future_target))
   model.add(LSTM(400, activation='relu', return_sequences=True))
   model.add(LSTM(400, activation='relu', return_sequences=True))
   model.add(LSTM(400, activation='relu', return_sequences=True))
   model.add(TimeDistributed(Dense(features)))
   model.compile(
       optimizer=opt, loss='mse', metrics=['mean_absolute_error']
   )

   # Train.
   model.fit(
       x_data, y_data, epochs=epochs, batch_size=10
   )

   # Predict.
   future_state = model.predict(initial_conditions)
   # Invert normalisation, and flattening.
   future_state = inverse_prepossessing(future_state)

The model consists of four LSTM layers, which in combination are able to
produce a more accurate and reliable prediction than a single LSTM
layer. As is evident, the activation function for each LSTM is the
rectified linear activation function, which is built into Keras. The
number of epochs can be specified by the end user depending on the
computational resources they have and what they need. More epochs will
evidently lead to a more accurate neural network.

An epoch is one forward pass and one backward pass of all the training
examples.

.. _noaa_initial_conditions:

Initial Conditions
==================

Global Data Assimilation System
-------------------------------

The initial conditions utilised by the software are from the Global Data
Assimilation System (GDAS), which is provided by the National Oceanic
and Atmospheric Adminstration in the United States. The primary reason
for utilising this data is that it is freely available to the general
public. In an ideal world, data from the European Centre for
Medium-Range Weather Forecasts would be utilised, however unfortunately,
there data is not freely available to the general public. This
fundamentally violates the software’s open source principles (these
principles are discussed in chapter `[5] <#5>`__).

The GDAS is a model to place observations into a gridded model space for
the purpose of initialising weather forecasts with observed data. This
system is utilised by the National Center for Environmental Prediction
for such a purpose. GDAS adds the following types of observations to a
gridded, 3-D, model space: surface observations, balloon data, wind
profiler data, aircraft reports, buoy observations, radar observations,
and satellite observations :cite:`gdas`. The initial
conditions provided by the GDAS to the software have a vertical pressure
co-ordinate, or the vertical co-ordiante is pressure. This co-ordinate
system is known as isobaric co-ordinates.

Isobaric Co-Ordinates
---------------------

In coordinate systems applied to the earth, the vertical coordinate
describes position in the vertical direction (that is, parallel to the
force of effective gravity). In meteorology, pressure can be a more
convenient vertical coordinate than altitude. One reason is that until
recently, radiosondes, which are the primary means of gathering
observations of weather variables above the earth’s surface, measure and
reported pressure, temperature, and humidity, but not altitude, as they
rise through the atmosphere :cite:`isobar_i`.

A radiosonde is an instrument carried by balloon to various levels of
the atmosphere and transmitting measurements by radio.

Another reason is that on scales large enough for the hydrostatic
approximation to be valid, the pressure-gradient force in
the equations of motion becomes simpler and density no longer becomes an
explicit variable in the tendency equations :cite:`isobar_i`.
Thus,a given geopotential gradient implies the same geostrophic wind at
any height, whereas a given horizontal pressure gradient implies
different values of the geostrophic wind depending on the
density :cite:`isobar_ii`.

From the Global Data Assimilation System, three prognostic variables are
chosen: geopotential height, air temperature, and relative humidity.

Geopotential Height is the height above sea level of a pressure level.
For example, if a station reports that the 500 hPa height at its
location is 5600 m, it means that the level of the atmosphere over that
station at which the atmospheric pressure is 500 hPa is 5600 meters
above sea level.

Geophysical sciences such as meteorology often prefer to express the
horizontal pressure gradient force as the gradient of geopotential along
a constant-pressure surface, because then it has the properties of a
conservative force. For example, the primitive equations which weather
forecast models solve use hydrostatic pressure as a vertical coordinate,
and express the slopes of those pressure surfaces in terms of
geopotential height. As such, this will be a parameter of great
interest. From the aforementioned three selected parameters, any other
parameter that is needed in the software can be calculated, including
the wind. This will be discussed in greater depth in the next chapter.


