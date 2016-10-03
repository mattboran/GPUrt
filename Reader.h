#pragma once

#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef READER_H
#define READER_H

#define M_PI 3.14159265359f
#define EPSILON 0.00001f
#define XRES 320
#define YRES 240

#define SAMPLES 512


//3 types of materials used in the radiance() function. 
enum Refl_t { DIFF, SPEC, REFR };

//This is the core of the program, hence ray-tracer
struct Ray{
	float3 origin;
	float3 dir;
	//construct on gpu
	__device__ Ray(float3 o, float3 d) : origin(o), dir(d) { }
	//debugging - this method prints the info about the ray to console
	__device__ void print_info(){
		printf("Ray composed of Origin and Direction.\n");
		printf("Origin = (%.2f, %.2f, %.2f)\n", origin.x, origin.y, origin.z);
		printf("Dir = (%.2f, %.2f, %.2f)\n", dir.x, dir.y, dir.z);
	}
};

//This is a model of the camera that is used to generate rays through the viewing plane. We use the left-hand-pointing
//model of a camera with defined origin, target direction (normalized), up (normalized), and right (normalized)
//given these vectors and a width and height for the screen (in pixels), we can generate rays through the view plane
//and turn the camera easily.
struct Camera{
	float3 camera_position;
	float3 camera_direction;
	float3 camera_up;
	float3 camera_right;
	//Camera should be constructed on device
	__device__ Camera(float3 pos, float3 target, float3 up) :
		camera_position(pos), camera_direction(normalize(target - pos)), camera_up(normalize(up)), camera_right(normalize(up)){
		camera_right = cross(camera_direction, camera_up);
	}
	__device__ Camera(float3 pos, float3 dir, float3 up, float3 right) :
		camera_position(pos), camera_direction(dir), camera_up(up), camera_right(right) {}

	//This method returns a Ray object generated from i and j coordinates (0 through XRES and 0 through YRES)
	__device__ inline Ray computeCameraRay(int i, int j, curandState *randstate){
		//inverse Xres and YRES are used to produce a correct aspect ratio based on pixel values of
		//the render screen

		//this following snippet of code introduces an in-pixel modifier for the precise position of the ray through the 
		//image plane. It applies a tent filter to the modifier to get more pixels in the center of the pixel than on the outside.

		//I commented out the tent filter because it seems to produce a strange form of diagonal aliasing.
		//Depending on resulting output, I will figure out what to do here instead.
		float r1 = 2 * curand_uniform(randstate), dx;
		if (r1 < 1){
			dx = sqrtf(r1) - 1;
		}
		else{
			dx = 1 - sqrtf(2 - r1);
		}
		float r2 = 2 * curand_uniform(randstate), dy;
		if (r2 < 1){
			dy = sqrtf(r2) - 1;
		}
		else{
			dy = 1 - sqrtf(2 - r2);
		}

		float inv_yres = 1.f / YRES;

		float normalized_i = ((i + dx) / (float)XRES) - 0.5f;
		float normalized_j = ((j + dy) / (float)YRES) - 0.5f;


		float3 image_point = normalized_i * camera_right * (inv_yres * XRES) +
			normalized_j * camera_up +
			camera_position + camera_direction;
		float3 ray_direction = image_point - camera_position;
		return Ray(camera_position, normalize(ray_direction));

	}
};


//Sphere - primitive object defined by radius and center.
//All primitives also have emmission (light, a vector) and color (another vector)
//struct __declspec(align(64)) Sphere{
struct __declspec(align(64))Sphere {
	float rad;
	float3 cent, emit, col; //center, emission, color
	Refl_t refl; //material type
	//const Geom_t geomtype = SPH;

	//constructor is implict 

	//ray-sphere intersection, returns distance to intersection or 0 if no hit.
	//using ray equation: o + td
	//sphere equation x*x + y*y + z*z = r*r where r is radius and x, y, z are coordinates in 3d
	//the quadratic equation is solved. If discriminant > 0, we have hit
	__device__ float intersectSphere(const Ray &r) const{
		float3 op = cent - r.origin;
		float t;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad;
		if (disc < 0) //intersection occurs when discriminant > 0
			return 0.f;
		else
			disc = sqrtf(disc);
		t = b - disc;//find intersection closest, in front of origin along ray.
		if (t > EPSILON)
			return t;
		t = b + disc;
		if (t > EPSILON)
			return t;
		else return 0.f;
	}
};

//Triangles are 128-byte objects defined by 3 points in 3d space, 3 UV coordinates in 2d space, and the same material
//proprties that spheres use (emmission, color, surface material type)
struct Triangle{
	float3 v1, v2, v3; //triangle defined by 3 vertices
	float3 emit, col;
	Refl_t refl;
	//const Geom_t geomtype;
	__host__ Triangle(float3 x, float3 y, float3 z, float3 e, float3 c, Refl_t r) :
		v1(x), v2(y), v3(z), emit(e), col(c), refl(r) {}


	//Moller-Trumbore ray-triangle intersection.
	//consider storing triangles as 1 vertex and 2 edges for faster compute
	__device__ float intersectTri(const Ray& r) const{
		float3 edge1, edge2;
		float3 P, Q, T;
		float det, inv_det, u, v, t;
		edge1 = v2 - v1;
		edge2 = v3 - v1;

		//calculate determinant
		P = cross(r.dir, edge2);
		det = dot(edge1, P);
		if (fabs(det) < EPSILON)
			return 0.0f;
		inv_det = 1.f / det;

		//distance from V1 to ray origin
		T = r.origin - v1;
		//u parameter, test bound
		u = dot(T, P) * inv_det;
		if (u < 0.f || u > 1.f)
			return 0.0f;
		//v parameter, test bound
		Q = cross(T, edge1);
		v = dot(r.dir, Q) * inv_det;
		if (v < 0.f || u + v > 1.f)
			return 0.0f;
		t = dot(edge2, Q) * inv_det;

		if (t > EPSILON){//hit
			return t;
		}

		return 0.f;
	}

	//return the face normal of the triangle. Interpolate (later in the project)
	__device__ inline float3 get_Normal(const float3& hitpt){
		float3 edge1 = v2 - v1;
		float3 edge2 = v3 - v1;
		return cross(edge2, edge1);
	}

	//this method is used for debugging memory. It prints the info about the triangle to console.
	__device__ void print_info(){
		printf("Trignale made up of V1, V2, V3, Color, Emit, and REFL_T\n");
		printf("V1 = (%.2f, %.2f, %.2f)\n", v1.x, v1.y, v1.z);
		printf("V2 = (%.2f, %.2f, %.2f)\n", v2.x, v2.y, v2.z);
		printf("V3 = (%.2f, %.2f, %.2f)\n", v3.x, v3.y, v3.z);
		printf("Color = (%.2f, %.2f, %.2f)\n", col.x, col.y, col.z);
	}
};


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

//This function returns the largest value of x y and z from a float3
__device__ inline float getMax(float3 f){
	if (f.x > f.y)
		return fmax(f.x, f.z);
	if (f.y > f.x)
		return fmax(f.y, f.z);
	if (f.z > f.x)
		return fmax(f.z, f.x);
	else return f.x;
}

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