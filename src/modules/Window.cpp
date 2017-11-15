/*
 * Copyright (C) 2014 Dominic Ward <contactdominicward@gmail.com>
 *
 * This file is part of Loudness
 *
 * Loudness is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Loudness is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Loudness.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include "Window.h"

namespace loudness{

    Window::Window(
            const WindowType& windowType,
            const IntVec &length,
            bool periodic) 
        : Module("Window"),
          windowType_(windowType),
          length_(length),
          periodic_(periodic),
          normalisation_(ENERGY)
    {}

    Window::Window(const WindowType& windowType, int length, bool periodic) :
        Module("Window"),
        windowType_(windowType),
        periodic_(periodic),
        normalisation_(ENERGY)
    {
       length_.assign(1, length); 
    }

    Window::Window():
        Module("Window")
    {}
    
    Window::~Window()
    {
		cudaFree(device_window_);
		cudaFree(device_input);
		cudaFree(device_output);
		cudaFreeHost(input_buffer);
		cudaFreeHost(output_buffer);
    }
   
    bool Window::initializeInternal(const SignalBank &input)
    {
        LOUDNESS_ASSERT(input.getNSamples() == length_[0], 
                    name_ << ": Number of input samples does not equal the largest window size!");

        //number of windows
        nWindows_ = (int)length_.size();
        LOUDNESS_DEBUG(name_ << ": Number of windows = " << nWindows_);
        window_.resize(nWindows_);

        //Largest window should be the first
        largestWindowSize_ = length_[0];
        LOUDNESS_DEBUG(name_ << ": Largest window size = " << largestWindowSize_);

        //first window (largest) does not require a shift
        windowOffset_.push_back(0);

        //check if we are using multi windows on one input channel
        int nOutputChannels = input.getNChannels();
        if ((input.getNChannels()==1) && (nWindows_>1))
        {
            LOUDNESS_DEBUG(name_ << ": Using parallel windows");
            method_ = ONE_CHANNEL_MULTI_WINDOW;
            nOutputChannels = nWindows_;
            //if so, calculate the delay
            int alignmentSample = largestWindowSize_ / 2;
            LOUDNESS_DEBUG(name_ << ": Alignment sample = " << alignmentSample);
            for(int w=1; w<nWindows_; w++)
            {
                int thisCentreSample = length_[w] / 2;
                int thisWindowOffset = alignmentSample - thisCentreSample;
                windowOffset_.push_back(thisWindowOffset);
                LOUDNESS_DEBUG(name_ << ": Centre sample for window " << w << " = " << thisCentreSample);
                LOUDNESS_DEBUG(name_ << ": Offset for window " << w << " = " << thisWindowOffset);
            }
        }
        else if ((input.getNChannels() > 1) & (nWindows_ == 1))
        {
            method_ = MULTI_CHANNEL_ONE_WINDOW;
        }
        else
        {
            LOUDNESS_ASSERT(input.getNChannels() == nWindows_,
                    "Multiple channels but incorrect window specification.");
        }
        
        //generate the normalised window functions
		unsigned int winTotalElements = 0;
        for (int w = 0; w < nWindows_; w++)
        {
            window_[w].assign(length_[w],0.0);
            generateWindow(window_[w], windowType_, periodic_);
            normaliseWindow(window_[w], normalisation_);
			winTotalElements += length_[w];
            LOUDNESS_DEBUG(name_ << ": Length of window " << w << " = " << window_[w].size());
        }

        //initialise the output signal
        output_.initialize(input.getNSources(),
                           input.getNEars(),
                           nOutputChannels,
                           largestWindowSize_,
                           input.getFs());
        output_.setFrameRate(input.getFrameRate());

		// Prepare the CUDA cards
		size_t window_size = sizeof(double)*winTotalElements;
		cudaMalloc(&device_window_, window_size);
		unsigned int numInputStreams, numOutputStreams;
		switch (method_)
		{
			case ONE_CHANNEL_MULTI_WINDOW:
			{
				numInputStreams = 1;
				numOutputStreams = nWindows_;
				break;
			}
			case MULTI_CHANNEL_ONE_WINDOW:
			{
				numInputStreams = input.getNChannels();
				numOutputStreams = output_.getNChannels();
			}

		}
		numCUDAStreams = input.getNSources()*input.getNEars();
		transferInputSize = numInputStreams*input.getNSamples();
		transferOutputSize = numOutputStreams*input.getNSamples()
		cudaMalloc(&device_input, sizeof(double)*transferInputSize*numCUDAStreams);
		cudaMalloc(&device_output, sizeof(double)*transferOutputSize*numCUDAStreams);
		cudaMallocHost(&input_buffer, sizeof(double)*transferInputSize*numCUDAStreams);
		cudaMallocHost(&output_buffer, sizeof(double)*transferOutputSize*numCUDAStreams);
		cudaStreams = new cudaStream_t[numCUDAStreams];
		for (int n = 0; n < numCUDAStreams; n++)
		{
			cudaStreamCreate(&cudaStreams[n]);
		}
        return 1;
    }

    void Window::processInternal(const SignalBank &input)
    {
        switch (method_)
        {
            case ONE_CHANNEL_MULTI_WINDOW:
            {
                for (int src = 0; src < input.getNSources(); ++src)
                {
                    for(int ear = 0; ear < input.getNEars(); ++ear)
                    {
                        for(int w = 0; w < nWindows_; ++w)
                        {
                            const Real* inputSignal = input
                                                      .getSignalReadPointer(
                                                          src,
                                                          ear,
                                                          0,
                                                          windowOffset_[w]);
                            Real* outputSignal = output_
                                                 .getSignalWritePointer(
                                                     src,
                                                     ear,
                                                     w,
                                                     0);

                            for(int smp = 0; smp < length_[w]; ++smp)
                            {
                                outputSignal[smp] = window_[w][smp]
                                                    * inputSignal[smp];
                            }
                        }
                    }
                }
            }
            case MULTI_CHANNEL_ONE_WINDOW:
            {
                for (int src = 0; src < input.getNSources(); ++src)
                {
                    for (int ear = 0; ear < input.getNEars(); ++ear)
                    {
                        for (int chn = 0; chn < input.getNChannels(); ++chn)
                        {
                            const Real* inputSignal = input
                                                      .getSignalReadPointer(
                                                          src,
                                                          ear,
                                                          chn,
                                                          0);
                            Real* outputSignal = output_
                                                 .getSignalWritePointer(
                                                     src,
                                                     ear,
                                                     chn,
                                                     0);

                            for (int smp = 0; smp < length_[0]; ++smp)
                            {
                                outputSignal[smp] = window_[0][smp]
                                                    * inputSignal[smp];
                            }
                        }
                    }
                }
            }
        }
    }

    void Window::resetInternal()
    {
    }

    //Window functions:
    void Window::hann(RealVec &window, bool periodic)
    {
        unsigned int N = window.size();
        int denom = N-1;//produces zeros on both sides
        if(periodic)//Harris (1978) Eq 27b 
            denom = N;
        for(uint i=0; i<window.size(); i++)
        {
            window[i] = 0.5 - 0.5 * cos(2.0*PI*i/denom);
        }
    }

    void Window::generateWindow(RealVec &window, const WindowType& windowType, bool periodic)
    {
        switch (windowType_)
        {
            case HANN:
                hann(window, periodic);
                LOUDNESS_DEBUG(name_ << ": Using a Hann window.");
                break;
            default:
                hann(window, periodic);
                LOUDNESS_DEBUG(name_ << ": Using a Hann window.");
                break;
        }
    }

    void Window::setNormalisation(const Normalisation& normalisation)
    {
        normalisation_ = normalisation;
    }

    void Window::normaliseWindow(RealVec &window, const Normalisation& normalisation)
    {
        if (normalisation != NONE)
        {
            double x = 0.0;
            double sum = 0.0, sumSquares = 0.0;
            double normFactor = 1.0;
            uint wSize = window.size();
            for(uint i=0; i < wSize; i++)
            {
                 x = window[i];
                 sum += x;
                 sumSquares += x*x;
            }

            switch (normalisation)
            {
                case (ENERGY):
                    normFactor = sqrt(wSize/sumSquares);
                    LOUDNESS_DEBUG(name_ << ": Normalising for energy.");
                    break;
                case (AMPLITUDE):
                    normFactor = wSize/sum;
                    LOUDNESS_DEBUG(name_ << ": Normalising for amplitude.");
                    break;
                default:
                    normFactor = sqrt(wSize/sumSquares);
            }

            LOUDNESS_DEBUG(name_ << ": Normalising window using factor: " << normFactor);
            for(uint i=0; i < wSize; i++)
                window[i] *= normFactor;
        }
    }

}
