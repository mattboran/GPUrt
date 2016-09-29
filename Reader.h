#pragma once

#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#ifndef READER_H
#define READER_H

#define M_PI 3.14159265359f
#define EPSILON 0.00001f
#define XRES 320
#define YRES 240

#define SAMPLES 1024

//This struct is used to load triangles from .obj files. This will be passed via wrapper 
//to the CUDA portion to be loaded into texture memory
struct loadingTriangle{
	float3 v1, v2, v3;
	loadingTriangle(float3 _v1, float3 _v2, float3 _v3){
		v1 = make_float3(_v1.x, _v1.y, _v1.z);
		v2 = make_float3(_v2.x, _v2.y, _v2.z);
		v3 = make_float3(_v3.x, _v3.y, _v3.z);
	}
};

//clamp a float on [0, 1]
inline float clampf(float x){
	return x < 0.f ? 0.f : x > 1.f ? 1.f : x;
}
//this function converts a float on [0.f, 1.f] to int on [0, 255], gamma-corrected by sqrt 2.2 (standard)
inline int toInt(float x){
	return int(pow(clampf(x), 1 / 2.2) * 255 + .5);
}

__device__ inline float max_float(float a, float b){
	if (a > b){
		return a;
	}
	return b;
}
__device__ inline float min_float(float a, float b){
	if (a < b){
		return a;
	}
	return b;
}
#endif