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

#ifndef XRES
#define XRES 240
#endif
#ifndef YRES
#define YRES 160
#endif
#ifndef SAMPLES
#define SAMPLES 32
#endif

//forward declarations
extern void renderKernelWrapper(float3 *out_host, int numspheres, loadingTriangle* tri_list, int numtris,float3 * AABB);
extern void testKernelWrapper(float *out_host);
extern void loadMeshToMemory(loadingTriangle *tri_list, int numberoftris);
void check_mesh(int numberoftris, int check_from, int check_to);
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
			std::string group;
			for (int q = 0; q < 3; q++){
				bool has_UV = false;
				if (oneline[i + 1] == ' '){
					j = i + 1;
					group = oneline[i];
				}
				else{
					while (oneline[j] != ' ' && j < linesize){
						j++;
					}
					group = oneline.substr(i, j);
				}
				
				if (oneline[j] == ' '){
					j++;
				}
				i = j;
				k = 0;//k is the runner used to grab endpoint for substring of "group"
				int _i = 0;//_i is the index that grabs the start of an index group (i.e. 4/81/14 is group. _i is 0 (4), k is 0 ('4'), because index 1 contains '/'. That substring is saved and parsed.
				//Then k is incremented (to skip '/'). _i is set to k. now _i is 2 (8), k runs once, to index 3 (1). second substring is 81...etc
				if (group.size() == 1){
					face_indices.push_back(stoi(group));
				}
				else{
					while (group[k] != '/' && k < group.size() - 1){//only record vertex index, not normal index or texture index
						k++;
					}
					face_indices.push_back(stoi(group.substr(_i, k)));
				}
				
				k++;
				//now move on to the second number (if there is one).
				//if we have face/no uv/norm, the face takes the form a//b exanple: 1//4 2//4 3//1
				if (k <= group.size()){
					if (group[k] == '/'){
							has_UV = false;
							has_UVs = false;
					}
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
	printf("Read in succesful.\n");
	return face_indices.size() / 3;//face indices represents the physical face of the triangle (3 vertices, 3 uv's). 
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
			printf("Adding index %s, ", onegroup.substr(_i, _j));

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
			printf(" %s, ", onegroup.substr(_i, _j));
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
			printf("and %s\n ", onegroup.substr(_i, _j));
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
void transformMesh(std::vector <loadingTriangle> &tri_list, const float3 &trans, const float3 &scale){
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
}

//populateTriangles takes the list of vertices and UVS, and compiles a list of TriangleFaces (when I say list, I mean std::vector)
//	min and max are float3's that define the bounding box for that particular mesh
//	hasUVs is a bool that reflects if the obj file parsed had uv_indices or not
//	scale and translate are factors used in modifying the location/size of the mesh.
//populateTriangles returns an int representing the number of triangle faces we're working with.
//	NOTE: Scale is applied first, THEN TRANSLATE.
//	Scale is performed  using 0,0,0 as an origin.
int populateTriangles(const std::vector<float3> &vertex_list, const std::vector<float2> &uv_list, const std::vector<unsigned int> &f_indices, const std::vector<unsigned int> &uv_indices, std::vector<loadingTriangle> &tris, float3 &min, float3 &max, const float3 &translate, const float3 &scale, const bool &has_UVs){
	const int num_faces = f_indices.size() / 3;
	min = make_float3(99999999.f, 99999999.f, 9999999.f);
	max = make_float3(-99999999.f, -99999999.f, -9999999.f);
	float3  temp;

	for (int i = 0; i < num_faces; i++){
		//first, grab the indices for the vertices that make this face
		int v1 = f_indices[3 * i] - 1;
		int v2 = f_indices[3 * i + 1] - 1;
		int v3 = f_indices[3 * i + 2] - 1;
		int uv1 = 0;
		int uv2 = 0;
		int uv3 = 0;

		//also grab the UV indices, if they exist
		if (has_UVs){
			uv1 = uv_indices[3 * i] - 1;
			uv2 = uv_indices[3 * i + 1] - 1;
			uv3 = uv_indices[3 * i + 2] - 1;
		}

		//save these vertices to compare min/max for creation of bounding box 
		float3 _v1 = vertex_list[v1];
		float3 _v2 = vertex_list[v2];
		float3 _v3 = vertex_list[v3];

		float2 _uv1 = make_float2(0, 0);
		float2 _uv2 = make_float2(0, 0);
		float2 _uv3 = make_float2(0, 0);

		if (has_UVs){
			_uv1 = uv_list[uv1];
			_uv2 = uv_list[uv2];
			_uv3 = uv_list[uv3];
		}
		//scale
		_v1 *= scale;
		_v2 *= scale;
		_v3 *= scale;
		//translate
		_v1 += translate;
		_v2 += translate;
		_v3 += translate;

		//next, create thisTri with the approrpiate vertices
		loadingTriangle thisTri(_v1, _v2, _v3);

		//calculate bounding box
		float3 tempmin = min3(_v1, _v2, _v3);
		float3 tempmax = max3(_v1, _v2, _v3);
		if (tempmin.x < min.x)
			min.x = tempmin.x;
		if (tempmin.y < min.y)
			min.y = tempmin.y;
		if (tempmin.z < min.z)
			min.z = tempmin.z;
		if (tempmax.x > max.x)
			max.x = tempmax.x;
		if (tempmax.y > max.y)
			max.y = tempmax.y;
		if (tempmax.z > max.z)
			max.z = tempmax.z;

		//add thisTri to tris list
		tris.push_back(thisTri);

	}
	return tris.size();
}

int main()
{
	//these are the parameters needed to read an .OBJ file and store the triangle mesh data. 
	//currently only supports 1 mesh. Kinda hack-y
	std::vector<float3> vertex_list, normal_list;
	std::vector<float2> uv_list;
	std::vector<unsigned int> f_indices, uv_indices;
	std::vector<loadingTriangle> triangle_list;

	float3 min = make_float3(-99999999.9f, -99999999.9f, -99999999.9f);
	float3 max = make_float3(99999999999.9f, 99999999999.9f, 99999999999.9f);
	float3 scale = make_float3(1,1,1);
	float3 translate = make_float3(5,0.25,5);
	char* filename = "models/tinytest.obj";
	std::cout << filename << " being loaded. \n\n";
	bool has_uvs = false;
	//int numtris = loadOBJ(filename, vertex_list, normal_list, uv_list, f_indices, uv_indices, has_uvs);
	
	int numtris= loadObj_onlyFaces(filename, vertex_list, f_indices);


	if (numtris == populateTriangles(vertex_list, uv_list, f_indices, uv_indices, triangle_list, min, max, translate, scale, has_uvs=false)){
		std::cout << "Successfully loaded " << filename << " with " << numtris << " triangles\n";
		printf("Bounding box min (%.2f, %.2f, %.2f)\nmax(%.2f, %.2f, %.2f)\n", min.x, min.y, min.z, max.x, max.y, max.z);
	}
	else{
		std::cout << "Failed loading " << filename << "\n";
	}
	//now load the mesh to CUDA memory. Since we use vectors, we pass address of vector[0] as our pointer to triangle_list
	//loadMeshToMemory(&triangle_list[0], triangle_list.size());

	printf("GPUrt initialized. \nRender for %d samples started...\n", SAMPLES);

	cudaEvent_t start, stop;
	float time_elapsed;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start timer
	cudaEventRecord(start);
	//schedule GPU threads and launch kernel from host CPU
	float3* out_host = new float3[XRES*YRES];
	float3* AABB = new float3[2];
	AABB[0] = min;
	AABB[1] = max;
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