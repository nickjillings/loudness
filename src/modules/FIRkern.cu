/*
* Copyright (C) 2017 Nicholas Jillings
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

__global__ double fir_complex_mul(cuDoubleComplex* inputs, cufftComplex* hs, unsigned int N)
{
	unsigned int blockSize = blockDim.y*blockIdx.x;
	unsigned int bid = blockDim.x*blockSize;
	unsigned int lid = threadIDx.x + threadIdx.y*blockDim.x;
	unsigned int tid = lid + bid;
	__shared__ cuDoubleComplex buffers[512]; // 1024 double precisions, or 8KB per block

	cuDoubleComplex* s_inputs = &buffers[0];
	cuDoubleComplex* s_hs = &buffers[256];
	if (tid < N)
	{
		// Perform the syncloads
		s_inputs[lid] = inputs[tid];
		s_hs[lid] = hs[tid];
		cuDoubleComplex d;
		d.x = s_inputs[lid].x * s_hs[lid].x - s_inputs[lid].y*s_hs[lid].y;
		d.y = s_inputs[lid].x * s_hs[lid].y + s_inputs[lid].y*s_hs[lid].x;
		outputs[tid] = d;
	}
}

void FIR::processInternal(const SignalBank &input)
{
	int num_sources = input.getNSources();
	int num_ears = input.getNEars();
	int num_channels = input.getNChannels();
	int num_samples = input.getNSamles();
	for (int src = 0; src < num_sources; ++src)
	{
		for (int ear = 0; ear < num_ears; ++ear)
		{
			for (int chn = 0; chn < num_channels; ++chn)
			{
				const Real* inputSignal = input.getSignalReadPointer
				(src, ear, chn);
				// Copy to the scratch buffer
				memset(copy_buffer, 0x0, sizeof(double)*bCoefs_pad_order * 2);
				for (int i = 0; i < num_samples; i++)
				{
					copy_buffer[i*2] = inputSignal[i];
				}
				unsigned int n = src*num_ears*num_channels + ear*num_channels + chn;
				cudaMemset(&inputs[n*bCoefs_pad_order], 0x0, sizeof(cuDoubleComplex)*bCoefs_pad_order);
				cudaMemcpy(&inputs[n*bCoefs_pad_order], copy_buffer, sizeof(cuDoubleComplex)*bCoefs_pad_order, cudaMemcpyHostToDevice);
			}
		}
	}

	// Now perform the batch FFT on the inputs
	cufftExecZ2Z(FFT, inputs, inputs, CUFFT_FORWARD);

	// Perform the complex multiplication on the inputs and hs
	int num_threads = 256;
	int num_blocks = (bCoefs_pad_order*num_streams) / num_threads;
	fir_complex_mul << <num_blocks, num_threads >> > (inputs, hs, bCoefs_pad_order*num_streams);

	// Perform the batch IFFT on the convolved buffers
	cufftExecZ2Z(FFT, inputs, inputs, CUFFT_INVERSE);

	// Collect each block synchronously to compute the overlap-add portions
	for (int src = 0; src < num_sources; ++src)
	{
		for (int ear = 0; ear < num_ears; ++ear)
		{
			for (int chn = 0; chn < num_channels; ++chn)
			{
				const Real* inputSignal = input.getSignalReadPointer
				(src, ear, chn);
				Real* outputSignal = output_.getSignalWritePointer
				(src, ear, chn);
				Real* z = delayLine_.getSignalWritePointer
				(src, ear, chn);
				// Copy to the scratch buffer
				memset(copy_buffer, 0x0, sizeof(double)*bCoefs_pad_order * 2);
				unsigned int n = src*num_ears*num_channels + ear*num_channels + chn;
				cudaMemcpy(copy_buffer, &inputs[n*bCoefs_pad_order], sizeof(cuDoubleComplex)*bCoefs_pad_order, cudaMemcpyDeviceToHost);
				for (int i = 0; i < num_samples; i++)
				{
					outputSignal[i] = copy_buffer[i * 2] + z[i];
					z[i] = copy_buffer[i * 2 + num_samples * 2];
				}
			}
		}
	}
}