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

/* This sample is a templatized version of the template project.
 * It also shows how to correctly templatize dynamically allocated shared
 * memory arrays.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "sharedmem.cuh"

#define		IPL				4// 1 2 4 8

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void
testKernel( T* g_idata, T* g_odata) 
{
	int block = blockIdx.x + blockIdx.y * gridDim.x;
	int index = threadIdx.x + IPL*block * blockDim.x;
#if 1
	T a[IPL];
//#pragma unroll
	for(int i=0; i<IPL; i++)
		a[i] = g_idata[index+i * blockDim.x];

//#pragma unroll
	for(int i=0; i<IPL; i++)
		g_odata[index+i * blockDim.x] = a[i];
	
#else
	T a0 = g_idata[index];
	T a1 = g_idata[index+blockDim.x];
	
	g_odata[index] = a0;
	g_odata[index+blockDim.x] = a1;
#endif
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
