
#include <iostream>
#include "Reader.h"
#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>


//forward declarations
uint hash(uint seed);

//These are the device sphere and triangle pointers. 
Sphere *dev_sphere_ptr;
Triangle *dev_tri_ptr;
//These two variables are the device pointers to min and max of AABB
float3 *dev_AABB_ptr;

//These numbers come directly from smallPT
//had to scale everything down by a factor of 10 to reduce artifacts.
//all spheres go in this list, here. This is messy. 
//spheres and triangles will eventually be moved to the .cpp file, and used through
//pointers in the .cu file
Sphere spheres[] = {
	{ 1e4f, { 1e4f + .10f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { 0.25f, 0.75f, 0.25f }, DIFF }, //Left 
	{ 1e4f, { -1e4f + 9.90f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
	{ 1e4f, { 5.00f, 4.08f, 1e4f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	{ 1e3f, { 5.00f, 4.08f, -1e4f + 60.00f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Front 
	{ 1e4f, { 5.00f, 1e4f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
	{ 1e4f, { 5.00f, -1e4f + 8.16f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	{ 0.5f, { 2.0f, 0.5f, 4.70f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
	{ 1.65f, { 7.30f, 1.65f, 7.80f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
	{ 60.0f, { 5.00f, 68.16f - .05f, 8.16f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

//LOADING DATA TO DEVICE DRAM////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//From here down, there are a series of methods that are called to load various things to 
//Device DRAM. This includes spheres, triangles, meshes, and AABB's. It also includes code
//That allows for storing the data in a different kind of device RAM (ie. texture memory, which
//is cached.
//////////////////////////////////////////////////////////////////////////////////////////

//this function loads the spheres defined above into DRAM
void loadSpheresToMemory(Sphere *sph_list, int numberofspheres){
	size_t numspheres = numberofspheres * sizeof(Sphere);
	printf("\mLoading %d bytes for %d spheres,\n", numspheres, numberofspheres);
	cudaMalloc((void **)&dev_sphere_ptr, numspheres);//void** cast is so cudaMalloc will accept the address of sphere pointer as parameter
	cudaMemcpy(dev_sphere_ptr, &sph_list[0], numspheres, cudaMemcpyHostToDevice);
}

//this function loads the triangles defined above into DRAM
void loadTrisToMemory(Triangle *tri_list, int numberoftris){
	size_t numtris = numberoftris * sizeof(Triangle);
	printf("Loading %d bytes for %d triangles,\n", numtris, numberoftris);
	cudaMalloc((void **)&dev_tri_ptr, numtris); //void** cast is so cudaMalloc will accept the address of triangle pointer as parameter
	cudaMemcpy(dev_tri_ptr, &tri_list[0], numtris, cudaMemcpyHostToDevice);
}

//this function loads an entire mesh's worth of triangles to dev_tri_ptr
void loadMeshToMemory(loadingTriangle *tri_list, int numberoftris){
	
	//I really hope this doesn't crash.This was done to avoid having the new Triangle[numberoftris] call because we didn't have a global 
	//constructor for Triangle.
	//Allocate memory for bytes that hold Triangle and that will be copied to dev_tri_ptr
	void* triangles = malloc((sizeof(Triangle)*numberoftris));
	//copy from tri_list to triangles
	memcpy(triangles, tri_list, sizeof(Triangle)*numberoftris);
	//our trianglelist is a pointer to the first element in triangles (i.e. our triangles!)
	Triangle* trianglelist = (Triangle*)triangles; 
	//This copies element by element from loadingTriangle into Triangle. It adds col, emit, and refl_t as hard-coded value. Eventally, this will be read
	//from .matl files. 
	for (int i = 0; i < numberoftris; i++){
		trianglelist[i].v1 = make_float3(tri_list[i].v1.x, tri_list[i].v1.y, tri_list[i].v1.z);
		trianglelist[i].v2 = make_float3(tri_list[i].v2.x, tri_list[i].v2.y, tri_list[i].v2.z);
		trianglelist[i].v3 = make_float3(tri_list[i].v3.x, tri_list[i].v3.y, tri_list[i].v3.z);
		trianglelist[i].col = make_float3(0.6,0.9,0.6);
		trianglelist[i].emit = make_float3(0, 0, 0);
		trianglelist[i].refl = DIFF;
	}
	printf("Loading mesh made of %d triangles for %d bytes\n\n", numberoftris, numberoftris*sizeof(Triangle));
	
	//Note - This will over-write the other triangles stored at &dev_tri_ptr
	size_t numtris = numberoftris * sizeof(Triangle);
	cudaMalloc((void**)&dev_tri_ptr, numtris);
	cudaMemcpy(dev_tri_ptr, &trianglelist[0], numtris, cudaMemcpyHostToDevice);
	delete[] trianglelist;
}

//this function loads the AABB to dev_min_ptr and dev_max_ptr
//with the bytes of data at &min and &max. 
//This cude is CLUSTERFUCKed. Casts on casts on casts 
void loadAABBtoMemory(float3 *AABB){
	size_t box_bytes = 2 * sizeof(float3);
	cudaMalloc((void**)&dev_AABB_ptr, box_bytes);
	cudaMemcpy(dev_AABB_ptr, &AABB[0], box_bytes, cudaMemcpyHostToDevice);
	printf("Successfully loaded AABB with:\nmin: (%.2f, %.2f, %.2f)\nmax: (%.2f, %.2f, %.2f)\n", AABB[0].x, AABB[0].y, AABB[0].z, AABB[1].x, AABB[1].y, AABB[1].z);
}

__device__ inline bool intersectBoundingBox(const Ray &r, float3* AABB){
	float3 invdir = make_float3(1.f / r.dir.x, 1.f / r.dir.y, 1.f / r.dir.z);

	float t1 = (AABB[0].x - r.origin.x) * invdir.x;
	float t2 = (AABB[1].x - r.origin.x) * invdir.x;
	float t3 = (AABB[0].y - r.origin.y) * invdir.y;
	float t4 = (AABB[1].y - r.origin.y) * invdir.y;
	float t5 = (AABB[0].z - r.origin.z) * invdir.z;
	float t6 = (AABB[1].z - r.origin.z) * invdir.z;

	float tmin = fmaxf(fmaxf(fminf(t1, t2),fminf(t3, t4)), fminf(t5, t6));
	float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
	//printf("min = (%.2f, %.2f, %.2f), max = (%.2f, %.2f, %.2f)\ntmin = %.2f, tmax=%.2f\n", AABB[0].x, AABB[0].y, AABB[0].z, AABB[1].x, AABB[1].y, AABB[1].z, tmin, tmax);
	//if tmax < 0, ray intersects AABB but in the inverse direction (i.e. it's behind us)
	if (tmax < 0)
	{
		//t = tmax;
		return false;
	}

	//if tmin  > tmax, ray doesn't intersect AABB
	if (tmin > tmax)
	{
		//t = tmax;
		return false;
	}
	//t = tmin;
	return true;
}
//This function is an inline implementation that intersects a list of triangles - tri_list . This intersect method goes through the device constant memory where
//the mesh is stored. It returns true if the ray intersects this entire mesh at all. 
__device__ inline void intersectListOfTriangles(const Ray &r, float &t, int &id, Triangle* tri_list, int numtris, int numspheres){
	float tprime = 1e15;
	for (int i = 0; i < numtris; i++){
		if ((tprime = tri_list[i].intersectTri(r)) && tprime < t){
			t = tprime;
			id = i + numspheres;
		}
	}
}

//World description: 9 spheres that form a modified Cornell box. this can be kept in const GPU memory (for now)
__device__ inline bool intersectScene(const Ray &r, float &t, int &id, Sphere *sphere_list, int numspheres, Triangle *tri_list, int numtris, float3 *AABB){
	float tprime;
	float inf = 1e15f;
	t = inf; //initialize t to infinite distance
	for (int i = 0; i < numspheres; i++){//cycle through all spheres, until i<0
		if ((tprime = sphere_list[i].intersectSphere(r)) && tprime < t){//new intersection is closer than previous closest
			t = tprime;
			id = i; //store hit sphere by ID (array index)
		}
	}
	//0 through 8 for ID represent spheres 1 through 9
	//the next ID's correspond to triangles
	//before testing all the triangles in the mesh, first test intersection with the bounding box defined by min and max (AABB[0], AABB[1]
	
	bool use_AABB = true;
	////this section of code calls inline functions that do the intersecting. This should makei  easier to add other *intersection modules* including using texture memory and 
	if (use_AABB){
		if (intersectBoundingBox(r, AABB)){
			intersectListOfTriangles(r, t, id, tri_list, numtris, numspheres);
		}
	}
	
	else{
		intersectListOfTriangles(r, t, id, tri_list, numtris, numspheres);
	}

	//if hit occured, t is > 0 and < inf.
	return t < inf;
}


//This function calculates radiance at a given ray, returned by a color
//This solves the rendering equation : outgoing radiance (pixel, point, w/e) = emitted radiance + reflected radiance
//reflected radiance is integral of incoming radiance from hemisphere above point about surface normal, multiplied
//by reflectence function of material, weighted by cosine incidence angle (Lambert's cosine law)
//Inputs: ray to calculate radiance along, and seeds for random num generation.
//Output: color at point.
__device__ float3 radiance(Ray &r, curandState *randstate, Sphere *sphere_list, Triangle *tri_list, int numtris, float3 *AABB){
	float3 accumulated = make_float3(0.f, 0.f, 0.f); //accumulated ray color for each iteration of loop
	float3 mask = make_float3(1.f, 1.f, 1.f);
	int numspheres = 9;

	Refl_t refltype;

	float3 hitpt;
	float3 norm;
	float3 hitnorm;
	float3 hitobj_color;
	float3 hitobj_emit;

	//ray bounce loop, will use Russian Roulette later
	for (int bounces = 0; bounces < 10; bounces++){ //this iterative loop replaces recursive path tracing method; max depth is 4 bounces (no RR)
		float t; //distance to hitt
		int id = 0; //index of hit 
		float3 d; //next ray direction

		if (!intersectScene(r, t, id, sphere_list, numspheres, tri_list, numtris, AABB))
			return make_float3(0.f, 0.f, 0.f); //return background color of 0 if no hit

		//if the loop gets to here, we have hit. compute hit point and surface normal
		if (id < numspheres){
			//identify which sphere was hit, calculate normal and transfer material properties
			//const Sphere &hitobj = sphere_list[id];
			hitpt = r.origin + r.dir*t;
			hitnorm = normalize(hitpt - sphere_list[id].cent); //surface normal
			//reverse normal if going through object - used to determine where we are for refraction
			float ntest = dot(hitnorm, r.dir);
			norm = (ntest < 0 ? hitnorm : hitnorm * -1);
			
			//material info
			hitobj_color = sphere_list[id].col;
			hitobj_emit = sphere_list[id].emit;

			accumulated += mask*hitobj_emit; //add emitted light to accumulated color, masked by previous bounces
			refltype = sphere_list[id].refl;
		}
		else{ //hit item was not a sphere, therefore it was a triangle.
			hitpt = r.origin + r.dir*t;
			hitnorm = tri_list[id - numspheres].get_Normal(hitpt);
			float ntest = dot(hitnorm, r.dir);
			norm = (ntest < 0 ? hitnorm : hitnorm * -1);

			//material info
			hitobj_color = tri_list[id - numspheres].col;
			hitobj_emit = tri_list[id - numspheres].emit;

			accumulated += mask*hitobj_emit; //add emitted light to accumulated color, masked by previous bounces
			refltype = tri_list[id - numspheres].refl;
		}

		//here we branch based on Refl_t; for now all are diffuse. Get a new random ray in hemisphere above hitnorm
		if (refltype == DIFF){
			//first create 2 random numbers
			float r1 = 2 * M_PI*curand_uniform(randstate); //random number on unit circle
			float r2 = curand_uniform(randstate); //random number for elevation 
			float r2sq = sqrtf(r2);

			//must get local orthonormal coordinates u v and w at hitpt for new random ray dir
			float3 w = norm;
			//based on abs.val of w's x component (> or < .1) cross w with straight along y (0,1,0) or along x(1,0,0)
			float3 u = normalize(cross((fabs(w.x)>0.1f ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f)), w));
			float3 v = cross(w, u);
			//now compute random ray direction on this hemisphere, in polar coordinates
			//note, cosine weighted importance sampling favors ray directions closer to the surf normal
			d = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
			//new ray origin is at hitpt, shifted a small amount along normal to prevent self-intersection


			mask *= hitobj_color; //multiply mask by object color for next bounce
			//apply Lambert's cosine law to get weighted importance sampling 
			mask *= dot(d, norm);
			hitpt += norm * .001f;
			mask *= 2.f; //divide by material pdf
		}
		else if (refltype == SPEC){//compute reflected ray direction
			d = reflect(r.dir, hitnorm);
			hitpt += norm * .001f;
			mask *= hitobj_color;
			//pdf = 1, don't need to divide by PDF
		}
		//REFR reflective type represents glass: index of refraction 1.4
		//Consider creating an index system for materials
		else {

			bool into = dot(norm, hitnorm) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(r.dir, norm);
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				d = reflect(r.dir, hitnorm); //d = r.dir - 2.0f * n * dot(n, r.dir);
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(r.dir * nnt - hitnorm * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, hitnorm));
				float Re = R0 + (1.f - R0) * c * c * c * c * c; //shlick's approx of Fresnel equation, probability of reflection
				float Tr = 1 - Re;
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < Re) // reflection ray
				{
					mask *= RP;
					d = reflect(r.dir, hitnorm);
				}
				else // transmission ray
				{
					mask *= TP;
					d = tdir;
				}
			}
		}
		//russian roulette
		if (bounces > 3){
			float P = getMax(mask);
			if (curand_uniform(randstate) > P)
				break;
			mask *= 1 / P;
		}
		r.origin = hitpt;
		r.dir = d;
	}

	return accumulated;
}

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
//this is the main rendering kernel that can be called from the CPU, runs in parallel on CUDA threads.
//each pixel runs in parallel
__global__ void render_kernel(float3 *out, uint hashedSampleNumber, Sphere *sphere_list, Triangle *tri_list, int numtris, float3 *AABB){
	//assign thread to every pixel
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("Pixeli, j = %d, %d\n", pixel_i, pixel_j);
	unsigned int i = (YRES - y - 1)*XRES + x; //get current pixel index from thread index
	
	curandState randstate;
	//int globalThreadID = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(x+y, 0, 0, &randstate);

	float3 cam_origin = make_float3(5.f, 5.2f, 19.56f);
	float3 cam_up = normalize(make_float3(0.f, 1, 0));
	float3 cam_target = make_float3(5.f, 5.2f, 0.f);
	Camera rayCaster = Camera(cam_origin, cam_target, cam_up);
	float3 col = make_float3(0.f, 0.f, 0.f); //reset for each pixel

	for (int s = 0; s < SAMPLES; s++){
		//primary ray dir, randomly jittered by a small amount (will be changed when there's a better camera struct)
		col = col + radiance(rayCaster.computeCameraRay(x, y, &randstate), &randstate, sphere_list, tri_list, numtris, AABB);// (1.f / SAMPLES);
	}
	//write rgb value of pixel to image buffer on GPU, clamped on [0.0f, 1.0f]
	float cor = (1.f / SAMPLES); //cor = correction: we want the average color
	out[i] = make_float3(clamp(col.x*cor, 0.f, 1.f), clamp(col.y*cor, 0.f, 1.f), clamp(col.z*cor, 0.f, 1.f));
}

//this wrapper function is used when the cpp main file calls the render kernel
void renderKernelWrapper(float3* out_host, int numspheres, loadingTriangle* tri_list, int numtris, float3* AABB){
	float3* out_dvc;

	cudaMalloc(&out_dvc, XRES * YRES * sizeof(float3));

	loadSpheresToMemory(spheres, numspheres);
	loadMeshToMemory(tri_list, numtris);
	loadAABBtoMemory(AABB);
	
	dim3 block(16, 16, 1);
	dim3 grid(XRES / block.x, YRES / block.y, 1);

	printf("\nLaunchng render_kernel for %d samples\n", SAMPLES);
	render_kernel << <grid, block >> > (out_dvc, hash(124), dev_sphere_ptr, dev_tri_ptr, numtris, dev_AABB_ptr);

	cudaMemcpy(out_host, out_dvc, XRES * YRES * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(out_dvc);

	cudaDeviceSynchronize();

}