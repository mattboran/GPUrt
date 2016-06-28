
#include <iostream>
#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#define M_PI 3.14159265359f
#define EPSILON 0.00001f
#define XRES 320
#define YRES 240
#define SAMPLES 128

//This is the core of the program, hence ray-tracer
struct Ray{
	float3 origin;
	float3 dir;
	//construct on gpu
	__device__ Ray(float3 o, float3 d) : origin(o), dir(d) { }
};
/*
struct Camera{
float3 lower_left;
float3 offset;
float3
};*/

//3 types of materials used in the radiance() function. 
enum Refl_t { DIFF, SPEC, REFR };
//3 types of geometry to be treated during radiance function. This causes big fork
enum Geom_t { SPH, TRI, BOX };

//Sphere - primitive object defined by radius and center.
//All primitives also have emmission (light, a vector) and color (another vector)
struct __declspec(align(64)) Sphere{
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

struct Triangle{
	float3 v1, v2, v3; //triangle defined by 3 vertices
	float3 emit, col;
	Refl_t refl;
	const Geom_t geomtype;
	__device__ Triangle(float3 x, float3 y, float3 z, float3 e, float3 c, Refl_t r) :
		v1(x), v2(y), v3(z), emit(e), col(c), refl(r), geomtype(TRI){}


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
		if (v < 0.f || v > 0.f)
			return 0.0f;
		t = dot(edge2, Q) * inv_det;

		if (t > EPSILON){//hit
			return t;
		}

		return 0.f;
	}
};

Sphere *dev_sphere_ptr;
float *dev_tri_ptr;

//These numbers come directly from smallPT
//had to scale everything down by a factor of 10 to reduce artifacts.
Sphere spheres[] = {
	{ 1e4f, { 1e4f + .10f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
	{ 1e4f, { -1e4f + 9.90f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
	{ 1e4f, { 5.00f, 4.08f, 1e4f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	{ 1e4f, { 5.00f, 4.08f, -1e4f + 60.00f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
	{ 1e4f, { 5.00f, 1e4f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
	{ 1e4f, { 5.00f, -1e4f + 8.16f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	{ 1.65f, { 2.70f, 1.65f, 4.70f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
	{ 1.65f, { 7.30f, 1.65f, 7.80f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
	{ 60.0f, { 5.00f, 68.16f - .05f, 8.16f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

void loadSpheresToMemory(Sphere *sph_list, int numberofspheres){

	size_t numspheres = numberofspheres * sizeof(Sphere);
	cudaMalloc((void **)&dev_sphere_ptr, numspheres);
	cudaMemcpy(dev_sphere_ptr, &sph_list[0], numspheres, cudaMemcpyHostToDevice);
	printf("%d numspheres\n", numspheres);

}

//World description: 9 spheres that form a modified Cornell box. this can be kept in const GPU memory (for now)
__device__ inline bool intersectScene(const Ray &r, float &t, int &id, Sphere *sphere_list){
	//float n = sizeof(spheres) / sizeof(Sphere); //get number of spheres by memory size
	float numspheres = sizeof(sphere_list) / sizeof(Sphere);
	numspheres = 9;
	float tprime;
	float inf = 1e15f;
	t = inf; //initialize t to infinite distance
	for (int i = int(numspheres); i >= 0; i--){//cycle through all spheres, until i<0
		if ((tprime = sphere_list[i].intersectSphere(r)) && tprime < t){//new intersection is closer than previous closest
			t = tprime;
			id = i; //store hit sphere by ID (array index)
		}
	}
	//if hit occured, t is > 0 and < inf.
	return t < inf;
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
//This function calculates radiance at a given ray, returned by a color
//This solves the rendering equation : outgoing radiance (pixel, point, w/e) = emitted radiance + reflected radiance
//reflected radiance is integral of incoming radiance from hemisphere above point about surface normal, multiplied
//by reflectence function of material, weighted by cosine incidence angle (Lambert's cosine law)
//Inputs: ray to calculate radiance along, and seeds for random num generation.
//Output: color at point.
__device__ float3 radiance(Ray &r, curandState *randstate, Sphere *sphere_list){
	float3 accumulated = make_float3(0.f, 0.f, 0.f); //accumulated ray color for each iteration of loop
	float3 mask = make_float3(1.f, 1.f, 1.f);

	//ray bounce loop, will use Russian Roulette later
	for (int bounces = 0; bounces < 20; bounces++){ //this iterative loop replaces recursive path tracing method; max depth is 4 bounces (no RR)


		float t; //distance to hitt
		int id = 0; //index of hit 
		float3 d; //next ray direction

		if (!intersectScene(r, t, id, sphere_list))
			return make_float3(0.f, 0.f, 0.f); //return background color of 0 if no hit

		//if the loop gets to here, we have hit. compute hit point and surface normal
		const Sphere &hitobj = sphere_list[id];
		float3 hitpt = r.origin + r.dir*t;
		float3 hitnorm = normalize(hitpt - hitobj.cent); //surface normal
		//reverse normal if going through object - used to determine where we are for refraction
		float ntest = dot(hitnorm, r.dir);
		float3 norm = (ntest < 0 ? hitnorm : hitnorm * -1);

		accumulated += mask*hitobj.emit; //add emitted light to accumulated color, masked by previous bounces
		Refl_t refltype = hitobj.refl;

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


			mask *= hitobj.col; //multiply mask by object color for next bounce
			//apply Lambert's cosine law to get weighted importance sampling 
			mask *= dot(d, norm);
			hitpt += norm * .001f;
			mask *= 2.f; //divide by material pdf
		}
		else if (refltype == SPEC){//compute reflected ray direction
			d = reflect(r.dir, hitnorm);
			hitpt += norm * .001f;
			mask *= hitobj.col;
			//pdf = 1, don't need to divide by PDF
		}
		//REFR reflective type represents glass: index of refraction 1.4
		//Consider creating an index system for materials
		else {// (refltype == REFR)

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


//this is the main rendering kernel that can be called from the CPU, runs in parallel on CUDA threads.
//each pixel runs in parallel
__global__ void render_kernel(float3 *out, uint hashedSampleNumber, Sphere *sphere_list){
	//assign thread to every pixel
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int i = (YRES - y - 1)*XRES + x; //get current pixel index from thread index
	
	curandState randstate;
	int globalThreadID = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(x+y + globalThreadID, 0, 0, &randstate);

	//hardcoded camera position - start from lower left corner of view plane
	Ray cam(make_float3(5.0f, 5.2f, 29.56f), normalize(make_float3(0.f, -0.0042612f, -.1)));
	//compute x and y offsets based on pixel coordinate
	float3 ofs_x = make_float3(XRES * .5135 / YRES, 0.f, 0.f);
	float3 ofs_y = normalize(cross(ofs_x, cam.dir))*0.5135f; //.5135 is field of view, roughly 30 degrees
	float3 col;
	col = make_float3(0.f, 0.f, 0.f); //reset for each pixel

	for (int s = 0; s < SAMPLES; s++){
		//primary ray dir, randomly jittered by a small amount (will be changed when there's a better camera struct)
		float3 dir = cam.dir + ofs_x * ((.25f + x) / XRES - 0.5f + curand_uniform(&randstate) / XRES) + ofs_y * ((.25f + y) / YRES - 0.5f + curand_uniform(&randstate) / YRES);
		//create incoming ray, add incoming radiance to final_col; push ray to start inside sphere that forms wall where we view from
		//that way, the scene does not distort at the edges
		col = col + radiance(Ray(cam.origin + dir * 4.f, normalize(dir)), &randstate, sphere_list) * (1.f / SAMPLES);
	}
	//write rgb value of pixel to image buffer on GPU, clamped on [0.0f, 1.0f]
	out[i] = make_float3(clamp(col.x, 0.f, 1.f), clamp(col.y, 0.f, 1.f), clamp(col.z, 0.f, 1.f));
}
inline float clampf(float x){
	return x < 0.f ? 0.f : x > 1.f ? 1.f : x;
}
//this function converts a float on [0.f, 1.f] to int on [0, 255], gamma-corrected by sqrt 2.2 (standard)
inline int toInt(float x){
	//return int(pow(x, 1 / 2.2) * 255.99f);
	return int(pow(clampf(x), 1 / 2.2) * 255 + .5);
}

int main()
{
	//StopWatchInterface *timer = NULL;
	float3* out_host = new float3[XRES * YRES]; //pointer to stored img on host memory
	float3* out_dvc; //pointer to image on GPU VRAM

	//allocate memory in GPU VRAM
	cudaMalloc(&out_dvc, XRES * YRES * sizeof(float3));

	//load spheres to memory
	loadSpheresToMemory(spheres, 9);

	//dim3 block and grid are CUDA specific for scheduling CUDA threads of stream processors
	dim3 block(16, 16, 1); //256 threads per block
	dim3 grid(XRES / block.x, YRES / block.y, 1);//number of blocks

	printf("CUDA initialized. \nRender for %d samples started...\n", SAMPLES);

	cudaEvent_t start, stop;
	float time_elapsed;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start timer
	cudaEventRecord(start);
	//schedule GPU threads and launch kernel from host CPU
	render_kernel << < grid, block >> > (out_dvc, hash(124), dev_sphere_ptr);

	//copy result back to host and free cuda memory
	cudaMemcpy(out_host, out_dvc, XRES * YRES * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(out_dvc);
	//end timer
	cudaEventRecord(stop);
	//double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
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
	return 0;
}