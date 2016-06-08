//Matt Boran's GPU Ray/Path Tracer
//this project is based on Kevin Beason's smallPT, a path tracer written in 99 lines of c++ in 2008
//further references and acknowledgements can be found on the 1st post of my blog 3dinhere.blogspot.com

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "cutil_math.h" //vector math is in here
//#include <ctime> //for execution time counter
#include <curand.h>//random number generator
#include <curand_kernel.h>//random number generator

#define M_PI 3.14159265359f
#define XRES 640
#define YRES 329
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
//3 types of materials used in the radiance() function. For now, we only use diffuse
enum Refl_t { DIFF, SPEC, REFR };

//Sphere - primitive object defined by radius and center.
//All primitives also have emmission (light, a vector) and color (another vector)
struct Sphere{
	float rad;
	float3 cent, emit, col; //center, emission, color
	Refl_t refl; //material type
	//constructor - on GPU
	//__device__ Sphere(float rad_, float3 cent_, float3 emit_, float3 col_, Refl_t refl_) :
		//rad(rad_), cent(cent_), emit(emit_), col(col_), refl(refl_) {}

	//ray-sphere intersection, returns distance to intersection or 0 if no hit.
	//using ray equation: o + td
	//sphere equation x*x + y*y + z*z = r*r where r is radius and x, y, z are coordinates in 3d
	//the quadratic equation is solved. If discriminant > 0, we have hit
	__device__ float intersectSphere(const Ray &r) const{
		float3 op = cent - r.origin;
		float t;
		float eps = 0.00001f; //eps is epsilon, used to disqualify self-intersections caused by floating point variables
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad;
		if (disc < 0) //intersection occurs when discriminant > 0
			return 0.f;
		else
			disc = sqrtf(disc);
		t = b - disc;//find intersection closest, in front of origin along ray.
		if (t > eps)
			return t;
		t = b + disc;
		if (t > eps)
			return t;
		else return 0.f;	
	}
};

//These numbers come directly from smallPT
/*
__constant__ Sphere spheres[] = {
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
	{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
	{ 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
	{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
	{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};*/

//had to scale everything down by a factor of 10 to reduce artifacts.
__constant__ Sphere spheres[] = {
	{ 1e4f, { 1e4f + .10f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
	{ 1e4f, { -1e4f + 9.90f, 4.08f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
	{ 1e4f, { 5.00f, 4.08f, 1e4f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	{ 1e4f, { 5.00f, 4.08f, -1e4f + 60.00f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
	{ 1e4f, { 5.00f, 1e4f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
	{ 1e4f, { 5.00f, -1e4f + 8.16f, 8.16f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	{ 1.65f, { 2.70f, 1.65f, 4.70f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
	{ 1.65f, { 7.30f, 1.65f, 7.80f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
	{ 60.0f, { 5.00f, 68.16f - .077f, 8.16f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};
//World description: 9 spheres that form a modified Cornell box. this can be kept in const GPU memory (for now)
__device__ inline bool intersectScene(const Ray &r, float &t, int &id){
	float n = sizeof(spheres) / sizeof(Sphere); //get number of spheres by memory size
	float tprime;
	float inf = 1e15f;
	t = inf; //initialize t to infinite distance
	for (int i = int(n); i>=0;i--){//cycle through all spheres, until i<0
		if ((tprime = spheres[i].intersectSphere(r)) && tprime < t){//new intersection is closer than previous closest
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

//This function calculates radiance at a given ray, returned by a color
//This solves the rendering equation : outgoing radiance (pixel, point, w/e) = emitted radiance + reflected radiance
//reflected radiance is integral of incoming radiance from hemisphere above point about surface normal, multiplied
//by reflectence function of material, weighted by cosine incidence angle (Lambert's cosine law)
//Inputs: ray to calculate radiance along, and seeds for random num generation.
//Output: color at point.
__device__ float3 radiance(Ray &r, curandState *randstate){
	float3 accumulated = make_float3(0.f, 0.f, 0.f); //accumulated ray color for each iteration of loop
	float3 mask = make_float3(1.f, 1.f, 1.f);

	//ray bounce loop, will use Russian Roulette later
	for (int bounces = 0; bounces < 4; bounces++){ //this iterative loop replaces recursive path tracing method; max depth is 4 bounces (no RR)
		float t; //distance to hitt
		int id = 0; //index of hit 
		if (!intersectScene(r, t, id))
			return make_float3(0.f, 0.f, 0.f); //return background color of 0 if no hit
		//if the loop gets to here, we have hit. compute hit point and surface normal
		const Sphere &hitobj = spheres[id];
		float3 hitpt = r.origin + r.dir*t;
		float3 hitnorm = normalize(hitpt - hitobj.cent); //surface normal
		//reverse normal if going through object - used to determine where we are for refraction
		float ntest = dot(hitnorm, r.dir);
		float3 norm = (ntest < 0 ? hitnorm : hitnorm * -1); 

		accumulated += mask*hitobj.emit; //add emitted light to accumulated color, masked by previous bounces

		//here we branch based on Refl_t; for now all are diffuse. Get a new random ray in hemisphere above hitnorm
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
		float3 d = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
		//new ray origin is at hitpt, shifted a small amount along normal to prevent self-intersection
		r.origin = hitpt + norm * 0.01f;
		r.dir = d; //the random dir we computed before

		mask *= hitobj.col; //multiply mask by object color for next bounce
		//apply Lambert's cosine law to get weighted importance sampling 
		mask *= dot(d, norm);
		mask *= 2.f; //fudge
	}
	return accumulated;
}

//this is the main rendering kernel that can be called from the CPU, runs in parallel on CUDA threads.
//each pixel runs in parallel
__global__ void render_kernel(float3 *out, uint hashedSampleNum){
	//assign thread to every pixel
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int i = (YRES - y - 1)*XRES + x; //get current pixel index from thread index
	//use pixel index as seeds
	unsigned int s1 = x;
	unsigned int s2 = y;

	curandState randstate;
	int globalThreadID = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(hashedSampleNum + globalThreadID, 0, 0, &randstate);

	//hardcoded camera position - start from lower left corner of view plane
	Ray cam(make_float3(5.0f, 5.2f, 29.56f), normalize(make_float3(0.f, -0.0042612f, -.1)));
	//compute x and y offsets based on pixel coordinate
	float3 ofs_x = make_float3(XRES * .5135 / YRES, 0.f, 0.f);
	float3 ofs_y = normalize(cross(ofs_x, cam.dir))*0.5135f; //.5135 is field of view, roughly 30 degrees
	float3 col;
	col = make_float3(0.f, 0.f, 0.f); //reset for each pixel

	for (int s = 0; s < SAMPLES; s++){
		//primary ray dir, randomly jittered by a small amount (will be changed when there's a better camera struct)
		float3 dir = cam.dir + ofs_x * ((.25f + x) / XRES - 0.5f + curand_uniform(&randstate)/XRES) + ofs_y * ((.25f + y) / YRES - 0.5f + curand_uniform(&randstate)/YRES);
		//create incoming ray, add incoming radiance to final_col; push ray to start inside sphere that forms wall where we view from
		//that way, the scene does not distort at the edges
		col = col + radiance(Ray(cam.origin + dir * 4.f, normalize(dir)), &randstate) * (1.f / SAMPLES);
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

	//dim3 block and grid are CUDA specific for scheduling CUDA threads of stream processors
	dim3 block(8, 8, 1); //64 threads per block
	dim3 grid(XRES / block.x, YRES / block.y, 1);//number of blocks

	printf("CUDA initialized. \nRender for %d samples started...\n", SAMPLES);

	//clock_t start_time = clock();
	//schedule GPU threads and launch kernel from host CPU
	render_kernel << < grid, block >> > (out_dvc, hash(1234));

	//copy result back to host and free cuda memory
	cudaMemcpy(out_host, out_dvc, XRES * YRES * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(out_dvc);
	//clock_t end_time = clock();
	//double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
	printf("Finsihed rendering!\n");
	
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