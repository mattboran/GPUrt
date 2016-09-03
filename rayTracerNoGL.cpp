#include <iostream>
#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef XRES
#define XRES 320
#endif
#ifndef YRES
#define YRES 240
#endif
#ifndef SAMPLES
#define SAMPLES 32
#endif

//forward declarations
extern void renderKernelWrapper(float3 *out_host);
extern void testKernelWrapper(float *out_host);
struct Sphere;



//hashing function to get seed for curandDevice
//this fast hash method was developed by Thomas Wang
//this is used to re-seed curand every sample
uint hash(uint seed){
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

//clamp a float on [0, 1]
inline float clampf(float x){
	return x < 0.f ? 0.f : x > 1.f ? 1.f : x;
}

//this function converts a float on [0.f, 1.f] to int on [0, 255], gamma-corrected by sqrt 2.2 (standard)
inline int toInt(float x){
	return int(pow(clampf(x), 1 / 2.2) * 255 + .5);
}


int main()
{
	bool debug = false;
	if (debug){
		printf("Debug mode. Trying test triangle\n");

		float* out_host = new float[100 * 100];

		testKernelWrapper(out_host);
		printf("done\n");
		FILE *img = fopen("test.txt", "w");
		for (int i = 0; i < 100 * 100; i++){
			fprintf(img, "%.0f", out_host[i]);
			if (i % 100 == 0){
				fprintf(img, "\n");
			}
		}
		delete[] out_host;
		return 0;
	}
	if (!debug)
	{
		printf("CUDA initialized. \nRender for %d samples started...\n", SAMPLES);

		cudaEvent_t start, stop;
		float time_elapsed;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//start timer
		cudaEventRecord(start);
		//schedule GPU threads and launch kernel from host CPU
		float3* out_host = new float3[XRES*YRES];

		renderKernelWrapper(out_host);

		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		printf("Finsihed rendering!\n");
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_elapsed, start, stop);
		printf("Render took %.4f seconds\n", time_elapsed / 1000.f);

		FILE *img = fopen("render.ppm", "w");
		fprintf(img, "P3\n%d %d\n %d\n", XRES, YRES, 255);

		//loop over pixels, write RGB values
		for (int i = 0; i < XRES * YRES; i++){
			fprintf(img, "%d %d %d\n", toInt(out_host[i].x),
				toInt(out_host[i].y),
				toInt(out_host[i].z));
		}

		printf("Saved render image to 'render.ppm'\n");

		//release host memory
		delete[] out_host;
	}
	return 0;
}