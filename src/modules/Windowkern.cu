/*
* Copyright (C) 2017 Nicholas Jillings
*
* This file is an extension of Loudness, see original copyright below
*
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

__device__ void loudness_apply_window(double* x, double* y, double* w, unsigned int tid)
{
	y[tid] = w[tid] * x[tid];
}

__global__ void loudness_window_multi_window_smem(double* input_vec, double* output_vec, double* packed_windows, unsigned int *w_sizes, unsigned int num_windows, unsigned int len_vec)
{
	// input_vec -> The single channel input
	// output_vec -> The output for the various windows
	// packed_windows -> Nestled window vectors
	// w_sizes -> The size of each packed windo
	// num_windows -> The number of packed windows
	// len_vec -> The length of the input vector
	// If input_vec == output_vec, in-place transform
	
	// This assumes the size of the input_vector is equal to the number of threads each block launches
	extern __shared__ double smem[];
	smem[threadIdx.x] = input_vec[threadIdx.x];
	
	loudness_window_multi_window(smem, output_vec, packed_windows, w_sizes, num_windows, len_vec);
}

__global__ void loudness_window_multi_window(double* input_vec, double* output_vec, double* packed_windows, unsigned int *w_sizes, unsigned int num_windows, unsigned int len_vec)
{
	// input_vec -> The single channel input
	// output_vec -> The output for the various windows
	// packed_windows -> Nestled window vectors
	// w_sizes -> The size of each packed windo
	// num_windows -> The number of packed windows
	// len_vec -> The length of the input vector
	// If input_vec == output_vec, in-place transform
	unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	// Identify which window the thread belongs to
	unsigned int window_block_id = 0;
	unsigned int N = w_sizes[0];
	unsigned int alignmentSample = N / 2; //Largest window should be the first
	unsigned int windowOffset = 0; // First window has no shift
	unsigned int window_block_end = N;
	unsigned int window_block_start = 0;
	while (tid < window_block_start)
	{
		window_block_id++;
		N = w_sizes[window_block_id];
		windowOffset = alignmentSample - (N / 2);
		window_block_start = window_block_end;
		window_block_end += N;
	}

	// Get the thread ID for the window
	unsigned int wid = gtid - window_block_start;
	if (wid > N || wid > len_vec) {
		return; // Thread exists beyond the last entry, so must exit now to avoid illegal mem access. Ideally no threads should hit this!
	}

	loudness_apply_window(&input_vec[windowOffset], &output_vec[window_block_start], &packed_windows[window_block_start], wid);
}

__global__ void loudness_window_one_window_smem(double* input_vec, double* output_vec, double* w, unsigned int N, unsigned int num_vectors)
{
	// input_vec -> The single channel input
	// output_vec -> The output for the various windows
	// packed_windows -> Nestled window vectors
	// w_sizes -> The size of each packed windo
	// num_windows -> The number of packed windows
	// len_vec -> The length of the input vector
	// If input_vec == output_vec, in-place transform

	// This assumes the size of the w is equal to the number of threads each block launches

	extern __shared__ double smem[];
	smem[threadIdx.x] = w[threadIdx.x];

	loudness_window_one_window(input_vec, output_vec, smem, N, num_vectors);
}

__global__ void loudness_window_one_window(double* input_vec, double* output_vec, double* w, unsigned int N, unsigned int num_vectors)
{
	// input_vec -> The single channel input
	// output_vec -> The output for the various windows
	// w -> The window to apply
	// N -> Size of window and each vector
	// num_vectors -> Number of vectors in input_vec and output_vec
	// If input_vec == output_vec, in-place transform
	unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	// Get the thread ID for the input_vector
	unsigned int v_num = gtid / N;
	unsigned int vid = gtid % N:
	if (v_num > num_vectors)
	{
		return; // Thread exists beyond the last entry
	}
	loudness_apply_window(&input_vec[v_num*N], &output_vec[v_num*N], w, vid);
}

void Window::processOneChannelMultiWindow(const SignalBack &input)
{

	for (int src = 0; src < input.getNSources(); ++src)
	{
		for (int ear = 0; ear < input.getNEars(); ++ear)
		{
			unsigned int cudaStreamId = src*input.getNEars() + ear;
			unsigned int inputOffset = cudaStreamId*transferInputSize;
			unsigned int outputOffset = cudaStreamId*transferOutputSize;
			const Real* inputSignal = input
				.getSignalReadPointer(
					src,
					ear,
					0,
					0);
			for (int n = 0; n < input.getNSamples(); n++)
			{
				input_buffer[n + inputOffset] = inputSignal[n];
			}
			cudaMemcpyAsync(&device_input[inputOffset], &input_buffer[inputOffset], sizeof(double)*transferInputSize, cudaMemcpyHostToDevice, cudaStreams[cudaStreamId]);
			if (input.getNSamples() <= 1024)
			{
				int num_threads = input.getNSamples();
				int num_blocks = transferOutputSize / num_threads;
				if (num_blocks*num_threads != transferOutputSize) {
					num_blocks += 1;
				}
				loudness_window_multi_window_smem <<<num_blocks, num_threads, sizeof(double)*num_threads, cudaStream[cudaStreamId] >>> (&device_input[inputOffset], &device_output[inputOffset], device_window_);
			}
			else {
				int num_threads = 512;
				int num_blocks = transferOutputSize / num_threads;
				if (num_blocks*num_threads != transferOutputSize) {
					num_blocks += 1;
				}
				loudness_window_multi_window << <num_blocks, num_threads, 0, cudaStream[cudaStreamId] >> > (&device_input[inputOffset], &device_output[inputOffset], device_window_);
			}
			cudaMemcpyAsync(&output_buffer[outputOffset], &device_output[outputOffset], sizeof(double)*transferOutputSize, cudaMemcpyDeviceToHost, cudaStreams[cudaStreamId]);
		}
	}

	for (int src = 0; src < input.getNSources(); ++src)
	{
		for (int ear = 0; ear < input.getNEars(); ++ear)
		{
			unsigned int cudaStreamId = src*input.getNEars() + ear;
			unsigned int outputOffset = cudaStreamId*transferOutputSize;
			cudaStreamSynchronize(cudaStreams[cudaStreamId]);
			for (int w = 0; w < nWindows_; ++w)
			{
				Real* outputSignal = output_
					.getSignalWritePointer(
						src,
						ear,
						w,
						0);
				for (int smp = 0; smp < length_[w]; ++smp)
				{
					outputSignal[smp] = output_buffer[smp + outputOffset];
				}
			}
		}
	}
}

void Window::processMultiChannelOneWindow(const SignalBack &input)
{
	for (int src = 0; src < input.getNSources(); ++src)
	{
		for (int ear = 0; ear < input.getNEars(); ++ear)
		{
			unsigned int cudaStreamId = src*input.getNEars() + ear;
			unsigned int inputOffset = cudaStreamId*transferInputSize;
			unsigned int outputOffset = cudaStreamId*transferOutputSize;
			for (int chn = 0; chn < input.getNChannels(); ++chn)
			{
				const Real* inputSignal = input
					.getSignalReadPointer(
						src,
						ear,
						chn,
						0);
				int num_samples = input.getNSamples();
				for (int smp = 0; smp < num_samples; ++smp)
				{
					input_buffer[smp + inputOffset + chn*num_samples] = inputSignal[smp];
				}
			}
			cudaMemcpyAsync(&device_input[inputOffset], &input_buffer[inputOffset], sizeof(double)*transferInputSize, cudaMemcpyHostToDevice, cudaStreams[cudaStreamId]);
			if (winTotalElements <= 1024)
			{
				int num_threads = winTotalElements;
				int num_blocks = transferOutputSize / num_threads;
				if (num_blocks*num_threads != transferOutputSize) {
					num_blocks += 1;
				}
				loudness_window_one_window_smem <<<num_blocks, num_threads, sizeof(double)*num_threads, cudaStreams[cudaStreamId] >>> (&device_input[inputOffset], &device_output[outputOffset], device_window_, input.getNSamples(), input.getNChannels());
			}
			else
			{
				int num_threads = 512;
				int num_blocks = transferOutputSize / num_threads;
				if (num_blocks*num_threads != transferOutputSize) {
					num_blocks += 1;
				}
				loudness_window_one_window <<<num_blocks, num_threads, 0, cudaStreams[cudaStreamId] >>> (&device_input[inputOffset], &device_output[outputOffset], device_window_, input.getNSamples(), input.getNChannels());
			}
			cudaMemcpyAsync(&output_buffer[outputOffset], &device_output[outputOffset], sizeof(double)*transferOutputSize, cudaMemcpyDeviceToHost, cudaStreams[cudaStreamId]);
		}
	}

	for (int src = 0; src < input.getNSources(); ++src)
	{
		for (int ear = 0; ear < input.getNEars(); ++ear)
		{
			unsigned int cudaStreamId = src*input.getNEars() + ear;
			unsigned int outputOffset = cudaStreamId*transferOutputSize;
			cudaStreamSynchronize(cudaStreams[cudaStreamId]);
			for (int chn = 0; chn < input.getNChannels(); ++chn)
			{
				Real* outputSignal = output_
					.getSignalWritePointer(
						src,
						ear,
						chn,
						0);
				for (int smp = 0; smp < outputSignal.getNSamples(); ++smp)
				{
					outputSignal[smp] = output_buffer[smp + outputOffset];
				}
			}
		}
	}
}

void Window::processInternal(const SignalBank &input)
{
	// Clear previous host_buffer
	memset(input_buffer, 0x0, transferInputSize*numCUDAStreams);
	memset(output_buffer, 0x0, transferOutputSize*numCUDAStreams);
	switch (method_)
	{
		case ONE_CHANNEL_MULTI_WINDOW:
		{
			processOneChannelMultiWindow(input);
			break;
		}
		case MULTI_CHANNEL_ONE_WINDOW:
		{
			processMultiChannelOneWindow(input);
			break;
		}
	}
}