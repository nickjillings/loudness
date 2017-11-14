# Loudness

Loudness is a C++ library with Python bindings for modelling perceived loudness. 
The library consists of processing modules which can be cascaded to form a loudness model.

This is an extension of the original library by Dominic Ward and can be found at [https://github.com/deeuu/loudness/issues](https://github.com/deeuu/loudness/issues).
This project aims to further accelerate the library using CUDA

## Dependencies

To build the C++ library you will need:
  - libsndfile1-dev >= 1.0.25
  - libfftw3-dev >= 3.3.3
  - zlib1g-dev >= 1.2.8
  - nvidia CUDA SDK >= 7.0

To build the Python bindings you will need:
  - swig >= 3.0.0
  - python-numpy-dev

## Note

This project is still in heavy development so is not stable. I am also now only
supporting Python 3.5+. Please register an issue at:
[https://github.com/deeuu/loudness/issues](https://github.com/deeuu/loudness/issues)

If you have CUDA issues, raise them at [https://github.com/nickjillings/loudness/issues](https://github.com/nickjillings/loudness/issues)

## Acknowledgments 

The library interface is based on the fantastic AIM-C:
https://code.google.com/p/aimc/

The cnpy library for reading numpy arrays in C++:
https://github.com/rogersce/cnpy

Ricard Marxer for the loudia audio project:
https://github.com/rikrd/loudia

### Example - Loudness of a 1 kHz tone @ 40 dB SPL according to ANSI S3.4:2007
~~~
import loudness as ln

# All inputs and outputs make use of a SignalBank
inputBank = ln.SignalBank()
nSources = 1
nEars = 1
nChannels = 1
nSamples = 1
fs = 1

# There are 4 dimensions
inputBank.initialize(nSources, nEars, nChannels, nSamples, fs)

# Set the centre frequency of the first channel
inputBank.setCentreFreq(0, 1000)

# Set the intensity in normalised units
level = 40
inputBank.setSample(0, 0, 0, 0, 10.0 ** (level / 10.0))

# The loudness model
model = ln.StationaryLoudnessANSIS342007()
model.initialize(inputBank)

# Now process the input
model.process(inputBank)

# Get the output of this loudness model
feature = 'InstantaneousLoudness'
outputBank = model.getOutput(feature)

print 'Loudness in sones %0.2f' % outputBank.getSample(0, 0, 0, 0)~~~
