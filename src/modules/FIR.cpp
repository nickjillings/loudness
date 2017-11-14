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

#include "FIR.h"

namespace loudness{

    FIR::FIR() : Module("FIR") {}

    FIR::FIR(const RealVec &bCoefs) :
        Module("FIR")
    {
        setBCoefs(bCoefs);
    }

    FIR::~FIR() {
		cudaFree(inputs);
		cudaFree(hs);
		cufftDestroy(FFT);
		free(copy_buffer);
	}

    bool FIR::initializeInternal(const SignalBank &input)
    {
        LOUDNESS_ASSERT(bCoefs_.size() > 0, name_ << ": No filter coefficients");

        //constants, order currently fixed for all ears/channels
        order_ = (int)bCoefs_.size() - 1;
        orderMinus1_ = order_ - 1;
        LOUDNESS_DEBUG("FIR: Filter order is: " << order_);
		// Prepare GPU
		// Padd the bCoefs to a power-of-2 and zero-pad x2
		unsigned int K = floor(log2(order_ + 1));
		if (1 << K < order_) {
			K++;
		}
		bCoefs_pad_order = 1 << (K + 1);
		double* bCoefs_mem = (double*)malloc(sizeof(double) * bCoefs_pad_order * 2);
		memset(bCoefs_mem, 0x0, sizeof(double) * bCoefs_pad_order * 2);
		for (int n = 0; n < order_; n++)
		{
			bCoefs_mem[n * 2] = bCoefs_[n];
		}
		// Create space on-card for the impulses (hs);
		size_t hs_size = sizeof(cuDoubleComplex)*bCoefs_pad_order;
		num_streams = input.getNSources()*input.getNEars()*input.getNChannels();
		cudaMalloc(&hs, hs_size*num_streams);

		// Create a use-once pair of memory for the first impulse transforms!
		double *h_in, *h_out;
		cudaMalloc(&h_in, sizeof(cuDoubleComplex)*bCoefs_pad_order);
		cudaMalloc(&h_out, sizeof(cuDoubleComplex)*bCoefs_pad_order);
		// Copy the strided memory pattern into h_in
		cudaMemcpy(&h_in, bCoefs_mem, sizeof(cuDoubleComplex)*bCoefs_pad_order, cudaMemcpyHostToDevice);
		// Plan and execute the FFT
		cufftResult hs_FFT;
		cufftPlan1d(&hs_FFT, bCoefs_pad_order, CUFFT_Z2Z, 1);
		cufftExecZ2Z(hs_FFT, h_in, h_out, CUFFT_FORWARD);

		// Copy the data from h_out into the number of stream pairs
		for (int n = 0; n < num_streams; n++)
		{
			cudaMemcpy(&hs[n*bCoefs_pad_order], &h_out, hs_size, cudaMemcpyDeviceToDevice);
		}

		// Tidy up
		cudaFree(h_in);
		cudaFree(h_out);
		cufftDestroy(hs_FFT);
		copy_buffer = bCoefs_mem;
		memset(copy_buffer, 0x0, sizeof(double)*bCoefs_pad_order * 2);

		// Now allocate all the remaining buffers
		cudaMalloc(&inputs, hs_size*num_streams);

		// Plan the FFTs
		cufftPlan1d(&FFT, bCoefs_pad_order, CUFFT_Z2Z, num_streams);

        //internal delay line - single vector for all ears
        delayLine_.initialize (input.getNSources(),
                   input.getNEars(),
                   input.getNChannels(),
                   order_,
                   input.getFs());

        //output SignalBank
        output_.initialize (input);

        return 1;
    }

    void FIR::resetInternal()
    {
        delayLine_.zeroSignals();
    }
}
