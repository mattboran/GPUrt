#pragma once

#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#ifndef READER_H
#define READER_H


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
#endif