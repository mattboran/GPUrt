#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Reader.h"
#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>


//forward declarations
extern void renderKernelWrapper(float3 *out_host, int numspheres, loadingTriangle* tri_list, int numtris,float3 * AABB);
extern void loadMeshToMemory(loadingTriangle *tri_list, int numberoftris);
inline float clampf(float x);
inline int toInt(float x);
uint hash(uint seed);

struct Sphere;
struct Ray;
struct Triangle;

//this function returns the minimum XYZ of the 3 vector3's presented.
//for bounding box creation
float3 min3(const float3 &v1, const float3 &v2, const float3 &v3){
	float3 min = make_float3(v1.x, v1.y, v1.z);
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
float3 max3(const float3 &v1, const float3 &v2, const float3 &v3) {
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

//This function loads a 3d model data from a .obj file.
//Note - this function strips normals and UVs. Only takes into account faces and vertices.
//Inputs: char array filename, vector- vertex list, vector - face index list, vector - loadingTriangles, array [2] of float3's for min and max (AABB)
//Output: int representing number of faces in face_list
int loadObj_onlyFaces(const char* filename, std::vector<float3> &vertex_list, std::vector<unsigned int> &face_indices, std::vector<loadingTriangle> &tri_list, float3 * AABB){
	//crash the whole program if we can't load the mesh.
	std::ifstream input(filename, std::ios::in);
	if (!input){
		std::cerr << "Cannot open " << filename << "\n";
		exit(1);
	}

	std::string oneline, onegroup;
	int i, j, _i, _j, linesize;
	float3 vertex;
	while (getline(input, oneline)){
		linesize = oneline.size();
		//
		//In .obj files, each vertex is indicated by a line starting with v followed by an x, y, and z value
		//
		//example: 
		//v 0.500000 0.687500 -0.093750
		//
		if (oneline[0] == 'v' && oneline[1] == ' '){
			i = 2;
			j = 0;
			//iterate to the end of the float value for x
			while (oneline[i + j] != ' '){
				j += 1;
			}//now oneline[j] = ' '
			vertex.x = stof(oneline.substr(i, j));
			//now iterate for y
			i += j + 1;
			j = 0;
			while (oneline[i + j] != ' '){
				j += 1;
			}
			vertex.y = stof(oneline.substr(i, j));
			//and iterate for z
			i += j + 1;
			j = linesize - i;
			vertex.z = stof(oneline.substr(i, j));
			//add vertex to list of vertices
			vertex_list.push_back(vertex);
		}
		//
		//Let's handle faces. We will only use the 1st set of numbers, before
		//any /'s. So we will ignore normals and UVs.
		//
		//f 1 3 5
		//f 1/2/3 3/3/4 5/3/3 should be read the same as the one above.
		//
		//To do this, we will first break down each line into face groups. Then we will find
		//the first instance of '/' and grab the string from index i to the '/'
		//That string will be  pushed into face_indices, and the other numbers after the / will be ignored
		//
		if (oneline[0] == 'f'){
			i = 2;
			j = 0;
			while (oneline[i + j] != ' '){
				j += 1;
			}
			onegroup = oneline.substr(i, j);

			//now, we need indexes local to onegroup. We will use _i and _j
			_i = 0;
			_j = 0;

			while (onegroup[_j] != '/' && _j < j){//j is the length of onegroup
				_j += 1;
			}
			face_indices.push_back(stoi(onegroup.substr(_i, _j)));

			i += j + 1;
			j = 0;
			while (oneline[i + j] != ' '){
				j += 1;
			}
			onegroup = oneline.substr(i, j);
			//now we do the 2nd group of indices, still only grabbing the first index of the group
			_i = 0;
			_j = 0;
			while (onegroup[_j] != '/' && _j < j){
				_j += 1;
			}
			face_indices.push_back(stoi(onegroup.substr(_i, _j)));
			//and the final group of face indices.
			i += j + 1;
			j = linesize - i;
			onegroup = oneline.substr(i, j); 
			_i = 0;
			_j = 0;

			while (onegroup[_j] != '/' && _j < j){
				_j += 1;
			}
			face_indices.push_back(stoi(onegroup.substr(_i, _j)));
		}

		
	}
	input.close();
	printf("Successfully read in .obj information from %s \n\n", filename);

	//Now comes the second part - this is the equivalent to populateTriangles. 
	//The goal is to load up the loadingTriangle vertices with the correct verts based on face_indices
	const int num_faces = face_indices.size() / 3;
	AABB[0] = make_float3(99999999.f, 99999999.f, 9999999.f);
	AABB[1] = make_float3(-99999999.f, -99999999.f, -9999999.f);
	//temporary vertices v1 through v3, as well as temp for AABB creation
	float3 v1, v2, v3, temp;

	for (int i = 0; i < num_faces; i++){
		//get the indices for the vertices that make up this face
		//i stands for index
		int i1 = face_indices[3 * i] - 1;
		int i2 = face_indices[3 * i + 1] - 1;
		int i3 = face_indices[3 * i + 2] - 1;

		 v1 = vertex_list[i1];
		 v2 = vertex_list[i2];
		 v3 = vertex_list[i3];

		 //create loadingTriangle with correct vertices
		 loadingTriangle thisTri(v1, v2, v3);
		

		 //calculate bounding box
		 temp = min3(v1, v2, v3);
		 
		 if (temp.x < AABB[0].x)
			 AABB[0].x = temp.x;
		 if (temp.y < AABB[0].y)
			 AABB[0].y = temp.y;
		 if (temp.z < AABB[0].z)
			 AABB[0].z = temp.z;

		 temp = max3(v1, v2, v3);
		 if (temp.x > AABB[1].x)
			 AABB[1].x = temp.x;
		 if (temp.y > AABB[1].y)
			 AABB[1].y = temp.y;
		 if (temp.z > AABB[1].z)
			 AABB[1].z = temp.z;

		 //add thisTri to tris list
		 tri_list.push_back(thisTri);
	}
	return tri_list.size();
}

//This function translates and scales a loadingTriangle
//Eventually, rotation will be implemented via some matrix classes I will implement. Eventually, if we are rotating large enough meshes, we will
//do it on-GPU
//This also transforms the AABB
void transformMesh(std::vector <loadingTriangle> &tri_list, const float3 &trans, const float3 &scale, float3 * AABB){
	//scale about origin
	for (int i = 0; i < tri_list.size(); i++){
		tri_list[i].v1 *= scale;
		tri_list[i].v2 *= scale;
		tri_list[i].v3 *= scale;
	}
	
	//translate
	for (int i = 0; i < tri_list.size(); i++){
		tri_list[i].v1 += trans;
		tri_list[i].v2 += trans;
		tri_list[i].v3 += trans;
	}
	AABB[0] *= scale;
	AABB[0] += trans;
	AABB[1] *= scale;
	AABB[1] += trans;
}

int main()
{
	//these are the parameters needed to read an .OBJ file and store the triangle mesh data. 
	//currently only supports 1 mesh. Kinda hack-y
	std::vector<float3> vertex_list;
	std::vector<unsigned int> f_indices;
	std::vector<loadingTriangle> triangle_list;

	float3 * AABB = new float3[2];
	AABB[0] = make_float3(-99999999.9f, -99999999.9f, -99999999.9f);
	AABB[1] = make_float3(99999999999.9f, 99999999999.9f, 99999999999.9f);

	float3 scale = make_float3(1,1,1);
	float3 translate = make_float3(5,0.05,4.75);

	char* filename = "models/teapot.obj";
	
	//int numtris = loadOBJ(filename, vertex_list, normal_list, uv_list, f_indices, uv_indices, has_uvs);
	
	int numtris= loadObj_onlyFaces(filename, vertex_list, f_indices, triangle_list, AABB);
	printf("Succesfully loaded %s for %d triangles. In main().\n", filename, numtris);
	transformMesh(triangle_list, translate, scale, AABB);

	printf("\nGPUrt initializing. \nRender is for %d samples and resolution %d by %d\n", SAMPLES, XRES, YRES);

	cudaEvent_t start, stop;
	float time_elapsed;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start timer
	cudaEventRecord(start);
	//schedule GPU threads and launch kernel from host CPU
	float3* out_host = new float3[XRES*YRES];
	renderKernelWrapper(out_host, 9, &triangle_list[0], triangle_list.size(), AABB);

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
	
	return 0;
}