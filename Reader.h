#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
//this function returns the minimum XYZ of the 3 vector3's presented.
//for bounding box creation
float3 min3(const float3 &v1, const float3 &v2, const float3 &v3){
	float3 min(v1);
	if (v2.x < min.x){
		min.x = v2.x;
	}
	if (v3.x < min.x){
		min.x = v3.x;
	}
	if (v2.y < min.y){
		min.y = v2.y;
	}
	if (v3.y < min.y){
		min.y = v3.y;
	}
	if (v2.z < min.z){
		min.z = v2.z;
	}
	if (v3.z < min.z){
		min.z = v3.z;
	}
	return min;
}

//this function returns the max XYZ of the 3 vector3 parameters
//for bounding box creation
float3 max3(const float3 &v1, const float3 &v2, const float3 &v3){
	float3 max(v1);
	if (v2.x > max.x){
		max.x = v2.x;
	}
	if (v3.x > max.x){
		max.x = v3.x;
	}
	if (v2.y > max.y){
		max.y = v2.y;
	}
	else if (v3.y > max.y){
		max.y = v3.y;
	}
	if (v2.z > max.z){
		max.z = v2.z;
	}
	else if (v3.z > max.z){
		max.z = v3.z;
	}
	return max;
}

//This struct is used to load triangles from .obj files. This will be passed via wrapper 
//to the CUDA portion to be loaded into texture memory
struct loadingTriangle{
	float4 v1, e1, e2;
	loadingTriangle(float3 _v1, float3 _v2, float3 _v3){
		v1 = make_float4(_v1.x, _v1.y, _v1.z, 0.f);
		e1 = make_float4(_v2.x, _v2.y, _v2.z, 0.f )- make_float4(_v1.x, _v1.y, _v1.z, 0.f);
		e2 = make_float4(_v3.x, _v3.y, _v3.z, 0.f)  - make_float4(_v1.x, _v1.y, v1.z, 0.f);
	}
};