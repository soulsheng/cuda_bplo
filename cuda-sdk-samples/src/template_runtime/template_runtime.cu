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

/* Template project which demonstrates the basics on how to setup a project 
* example application, doesn't use cutil library.
*/

#include <stdio.h>
#include <string.h>
#include <iostream>

#include <shrQATest.h>
#include <vector_types.h>

using namespace std;

bool g_bQATest = false;

#define IPL	2

#define ALIGNED   1

#if ALIGNED
typedef int1 INT1;
#else
struct tagInt1{int x;};
typedef tagInt1  INT1;
#endif

#ifdef _WIN32
   #define STRCASECMP  _stricmp
   #define STRNCASECMP _strnicmp
#else
   #define STRCASECMP  strcasecmp
   #define STRNCASECMP strncasecmp
#endif

#define ASSERT(x, msg, retcode) \
    if (!(x)) \
    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
    }

__global__ void sequence_gpu(INT1 *d_ptr, int1 length)
{
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;
#if !(IPL-1)
	if (elemID<length)
	{
		d_ptr[elemID]=elemID;
	}
#else
	if (elemID+blockIdx.x * blockDim.x*(IPL-1)<length.x)
	{
#pragma unroll 
		for (int j=0;j<IPL;j++)
		{	
			INT1 value1;
			value1.x = elemID;
			d_ptr[elemID + j*blockIdx.x * blockDim.x] = value1;
		}
	}
#endif
}


void sequence_cpu(INT1 *h_ptr, int1 length)
{
    for (int elemID=0; elemID<length.x; elemID++)
    {
		INT1 value1;
		value1.x = elemID;
        h_ptr[elemID] = value1;
    }
}

void processArgs(int argc, char **argv)
{
    for (int i=1; i < argc; i++) {
        if((!STRNCASECMP((argv[i]+1), "noprompt", 8)) || (!STRNCASECMP((argv[i]+2), "noprompt", 8)) )
        {
            g_bQATest = true;
        }
    }
}

int main(int argc, char **argv)
{
	shrQAStart(argc, argv);

    cout << "CUDA Runtime API template" << endl;
    cout << "=========================" << endl;
    cout << "Self-test started" << endl;

    const int N = (1<<20); 

    processArgs(argc, argv);

    INT1 *d_ptr;
    ASSERT(cudaSuccess == cudaMalloc    (&d_ptr, N * sizeof(INT1)), "Device allocation of " << N << " ints failed", -1);

    INT1 *h_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(INT1)), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(256,1,1);
    dim3 cudaGridSize( (N + cudaBlockSize.x - 1) / cudaBlockSize.x /IPL, 1, 1);
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, make_int1(N) );
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, make_int1(N));

    cout << "CUDA and CPU algorithm implementations finished" << endl;

    INT1 *h_d_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N * sizeof(INT1)), "Host allocation of " << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMemcpy(h_d_ptr, d_ptr, N * sizeof(INT1), cudaMemcpyDeviceToHost), "Copy of " << N << " ints from device to host failed", -1);
    bool bValid = true;
    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i].x != h_d_ptr[i].x)
        {
            bValid = false;
        }
    }

    ASSERT(cudaSuccess == cudaFree(d_ptr),       "Device deallocation failed", -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed",   -1);

    cout << "Memory deallocated successfully" << endl;
    cout << "TEST Results " << endl;
    
    shrQAFinishExit(argc, (const char **)argv, (bValid ? QA_PASSED : QA_FAILED));
}
