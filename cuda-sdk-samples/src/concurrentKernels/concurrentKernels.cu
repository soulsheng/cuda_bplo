/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//
// This sample demonstrates the use of streams for concurrent execution. It also illustrates how to 
// introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function introduced 
// in CUDA 3.2.
//
// Devices of compute capability 1.x will run the kernels one after another
// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <stdio.h>
//#include <cutil_inline.h>
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrUtils.h>
#include <shrQATest.h>

const char *sSDKsample = "concurrentKernels";

// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block(clock_t* d_o, clock_t clock_count)
{ 
	clock_t start_clock = clock();
	
	clock_t clock_offset = 0;

	while( clock_offset < clock_count ) {
		clock_offset = clock() - start_clock;
	}

	d_o[0] = clock_offset;
}


// Single warp reduction kernel
__global__ void sum(clock_t* d_clocks, int N)
{
	__shared__ clock_t s_clocks[32];

	clock_t my_sum = 0;

	for( int i = threadIdx.x; i < N; i+= blockDim.x ) {
		my_sum += d_clocks[i];
	}

	s_clocks[threadIdx.x] = my_sum;
	syncthreads();	

	for( int i=16; i>0; i/=2) {
		if( threadIdx.x < i ) {
			s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
		}
		syncthreads();	
	}	

	d_clocks[0] = s_clocks[0];
}


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
	if(cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);        
	}
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
		exit(-1);
	}

	if (devID < 0)
		devID = 0;

	if (devID > deviceCount-1)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
		fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
		fprintf(stderr, "\n");
		return -devID;
	}

	cudaDeviceProp deviceProp;
	checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

	if (deviceProp.major < 1)
	{
		fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
		exit(-1);                                                  
	}

	checkCudaErrors( cudaSetDevice(devID) );
	printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

	return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
	int current_device     = 0, sm_per_multiproc  = 0;
	int max_compute_perf   = 0, max_perf_device   = 0;
	int device_count       = 0, best_SM_arch      = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount( &device_count );

	// Find the best major SM Architecture GPU device
	while (current_device < device_count)
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999)
		{
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{
			sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

		if( compute_perf  > max_compute_perf )
		{
			// If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 )
			{
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch)
				{
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
			else
			{
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv)
{
	cudaDeviceProp deviceProp;
	int devID = 0;
	// If the command-line has a device number specified, use it
	if (checkCmdLineFlag(argc, argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, argv, "device=");
		if (devID < 0)
		{
			printf("Invalid command line parameter\n ");
			exit(-1);
		}
		else
		{
			devID = gpuDeviceInit(devID);
			if (devID < 0)
			{
				printf("exiting...\n");
				shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
				exit(-1);
			}
		}
	}
	else
	{
		// Otherwise pick the device with highest Gflops/s
		devID = gpuGetMaxGflopsDeviceId();
		checkCudaErrors( cudaSetDevice( devID ) );
		checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	return devID;
}
// end of CUDA Helper Functions

int main(int argc, char **argv)
{
    int nkernels = 8;               // number of concurrent kernels
    int nstreams = nkernels + 1;    // use one more stream than concurrent kernel
    int nbytes = nkernels * sizeof(clock_t);   // number of data bytes
    float kernel_time = 10; // time the kernel should run in ms
    float elapsed_time;   // timing variables
    int cuda_device = 0;

    shrQAStart(argc, argv); 

    // get number of kernels if overridden on the command line
    if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
        nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
        nstreams = nkernels + 1;
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char**)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDevice(&cuda_device));	

    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (deviceProp.concurrentKernels == 0 )) {
        shrLog("> GPU does not support concurrent kernel execution\n");
        shrLog("  CUDA kernel runs will be serialized\n");
    }

    shrLog("> Detected Compute SM %d.%d hardware with %d multi-processors\n", 
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount); 

    // allocate host memory
    clock_t *a = 0;                     // pointer to the array data in host memory
    checkCudaErrors( cudaMallocHost((void**)&a, nbytes) ); 

    // allocate device memory
    clock_t *d_a = 0;             // pointers to data and init value in the device memory
    checkCudaErrors( cudaMalloc((void**)&d_a, nbytes) );

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++)
        checkCudaErrors( cudaStreamCreate(&(streams[i])) );

    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors( cudaEventCreate(&start_event) );
    checkCudaErrors( cudaEventCreate(&stop_event) );

   
    // the events are used for synchronization only and hence do not need to record timings
    // this also makes events not introduce global sync points when recorded which is critical to get overlap 
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t*) malloc(nkernels * sizeof(cudaEvent_t));
    for(int i = 0; i < nkernels; i++)
        checkCudaErrors( cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming) );

    //////////////////////////////////////////////////////////////////////
    // time execution with nkernels streams
    clock_t total_clocks = 0;
    clock_t time_clocks = kernel_time * deviceProp.clockRate;
	
    cudaEventRecord(start_event, 0);
    // queue nkernels in separate streams and record when they are done
    for( int i=0; i<nkernels; ++i)
    {
        clock_block<<<1,1,0,streams[i]>>>(&d_a[i], time_clocks );
        total_clocks += time_clocks;
        checkCudaErrors( cudaEventRecord(kernelEvent[i], streams[i]) );
	
        // make the last stream wait for the kernel event to be recorded
        checkCudaErrors( cudaStreamWaitEvent(streams[nstreams-1], kernelEvent[i],0) );
    }

    // queue a sum kernel and a copy back to host in the last stream. 
    // the commands in this stream get dispatched as soon as all the kernel events have been recorded
    sum<<<1,32,0,streams[nstreams-1]>>>(d_a, nkernels);
    checkCudaErrors( cudaMemcpyAsync(a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams-1]) );
 
    // at this point the CPU has dispatched all work for the GPU and can continue processing other tasks in parallel

    // in this sample we just wait until the GPU is done
    checkCudaErrors( cudaEventRecord(stop_event, 0) );
    checkCudaErrors( cudaEventSynchronize(stop_event) );
    checkCudaErrors( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    
    shrLog("Expected time for serial execution of %d kernels = %.3fs\n", nkernels, nkernels * kernel_time/1000.0f);
    shrLog("Expected time for concurrent execution of %d kernels = %.3fs\n", nkernels, kernel_time/1000.0f);
    shrLog("Measured time for sample = %.3fs\n", elapsed_time/1000.0f);

    bool bTestResult  = (a[0] > total_clocks);

    // release resources
    for(int i = 0; i < nkernels; i++) {
        cudaStreamDestroy(streams[i]); 
        cudaEventDestroy(kernelEvent[i]);
    }
    free(streams);
    free(kernelEvent);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(a);
    cudaFree(d_a);

    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (bTestResult) ? QA_PASSED : QA_FAILED);
}
