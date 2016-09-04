#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
extern void readMeshToTexMemory();
struct Sphere;

//This struct is used to load triangles from .obj files. This will be passed via wrapper 
//to the CUDA portion to be loaded into texture memory
struct loadingTriangle{
	float4 v1, e1, e2;
	loadingTriangle(float3 _v1, float3 _v2, float3 _v3){
		v1 = make_float4(_v1);
		e1 = make_float4(_v2 - _v1);
		e2 = make_float4(_v3 - _v1);
	}
};
//This function loads 3d model data from a .obj file.
//Inputs: char array filename, vector - list of all vertices, vector - list of all UV's, vector - list of all faces indices
//Output: int representing number of faces in face_list
//
//Currently, this ignores vertex normals, face normals, or materials. It only supports a single mesh at a time
int loadOBJ(const char* filename, std::vector<float3> &vertex_list, std::vector<float3> &norm_list, std::vector<float2> &uv_list, std::vector<unsigned int> &face_indices, std::vector<unsigned int> &uv_indices, bool &has_UVs){
	std::ifstream input(filename, std::ios::in);
	if (!input){
		std::cerr << "Cannot open " << filename << "\n";
		exit(1);
	}
	std::string oneline;
	has_UVs = true;
	float* xyz = new float[3];
	float* uv = new float[2];
	bool* face_index_format = new bool[3];
	//bool checked_face_format = false;

	int i, j; //i is the index of the character being parsed through the line
	//1. iterate through all lines of the file
	//2. if vertex, get coordinates and add to vertex_list
	//3. if UV, get coordinates and add to uv_list
	//4. if face index, split the line into the 3 groups of (f_index/uv_index/norm_index)
	//-  uses 2 additional index characters _i (for the start of substring) and k (end of substring)
	//-  these indices are iterated the same way that i and j are iterated to grab substrings from the entire line
	//4a. parse the f_index component, add to f_indices
	//4b. parse the uv_index component, add to uv_indices
	//4c. TODO, if necessary: parse the norm_index component, add to norm_index


	//Note: This code assumes that the .obj files uses strictly triangles. It will probably break if you feed it quads or n-gons
	while (getline(input, oneline)){
		i = 0, j = 0; //indices used to split string into substrings to parse coordinates/indices
		int linesize = oneline.size();
		//
		//In .obj files, each vertex is indicated by a line starting with v followed by an x, y, and z value
		//
		//example: 
		//v 0.500000 0.687500 -0.093750
		//
		if (oneline[0] == 'v' &&  oneline[1] == ' '){
			i = 2;
			j = 2;
			//cycle through x y and z coordinates for each vertex
			for (int q = 0; q < 3; q++){
				i = j;
				while (oneline[j] != ' ' && j < linesize){
					j++;
				}
				xyz[q] = stof(oneline.substr(i, j));
				j++;
			}
			float3 this_vert = make_float3(xyz[0], xyz[1], xyz[2]);
			vertex_list.push_back(this_vert);
		}
		//
		//.obj files store a normal at each vertex.
		//
		//example:
		//vn 0.500000 0.687500 -0.093750
		//
		if (oneline[0] == 'v' &&  oneline[1] == 'n'){
			i = 3;
			j = 3;
			//cycle through x y and z coordinates for each vertex
			for (int q = 0; q < 3; q++){
				i = j;
				while (oneline[j] != ' ' && j < linesize){
					j++;
				}
				xyz[q] = stof(oneline.substr(i, j));
				j++;
			}
			float3 this_norm = make_float3(xyz[0], xyz[1], xyz[2]);
			norm_list.push_back(this_norm);
		}
		//
		//Texture coordinates are the 3d vertices projected onto UV space.
		//
		//example:
		//vt -0.003000 0.100000
		//
		else if (oneline[0] == 'v' && oneline[1] == 't'){
			i = 3;
			j = 3;
			//cycle through u and v coordinates for each vertex
			for (int q = 0; q < 2; q++){
				while (oneline[j] != ' ' && j < linesize){
					j++;
				}
				uv[q] = stof(oneline.substr(i, j));
				j++;
				i = j;
			}
			float2 this_uv = make_float2(uv[0], uv[1]);
			uv_list.push_back(this_uv);
		}
		//
		//Faces in .obj files are stored as combinations of vertices, texture cordinates, and vertex normals
		//
		//f vertex_index/texture_index/vertex_normal
		//
		//example:
		//f 1/2/1 2/3/3 3/1/2
		//
		//However: texture index and vertex normal are not necessary, so the following is valid:
		//f 1 2 3 (face from vertices 1 2 and 3)
		//f 1//1 2//3 3//2 (face from vertices 1, 2, 3, with no texture coordinates, with normals 1 3 and 2

		else if (oneline[0] == 'f'){
			i = 2;//start position of substring (each of 3 face index groups)
			j = 2; //end position of substring (each of 3 face index groups)
			int k = 2; //used to divide string by token '/'

			for (int q = 0; q < 3; q++){
				bool has_UV = true;
				while (oneline[j] != ' ' && j < linesize){
					j++;
				}
				std::string group = oneline.substr(i, j);
				if (oneline[j] == ' '){
					j++;
				}
				i = j;
				k = 0;//k is the runner used to grab endpoint for substring of "group"
				int _i = 0;//_i is the index that grabs the start of an index group (i.e. 4/81/14 is group. _i is 0 (4), k is 0 ('4'), because index 1 contains '/'. That substring is saved and parsed.
				//Then k is incremented (to skip '/'). _i is set to k. now _i is 2 (8), k runs once, to index 3 (1). second substring is 81...etc
				while (group[k] != '/' && k < group.size()){//only record vertex index, not normal index or texture index
					k++;
				}
				face_indices.push_back(stoi(group.substr(_i, k)));
				k++;
				//now move on to the second number (if there is one).
				//if we have face/no uv/norm, the face takes the form a//b exanple: 1//4 2//4 3//1
				if (group[k] == '/' || k == group.size()){
					has_UV = false;
					has_UVs = false;
				}
				_i = k;//if, however, we do have a uv item, go to the next '/'
				while (group[k] != '/'&& k < group.size()){
					k++;
				}
				if (has_UV){//add that uv element to the uv_indices list if it exists
					uv_indices.push_back(stoi(group.substr(_i, k)));
				}
				k++;
				_i = k;


			}
		}

	}
	delete xyz;
	delete uv;
	return face_indices.size() / 3;//face indices represents the physical face of the triangle (3 vertices, 3 uv's). 
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