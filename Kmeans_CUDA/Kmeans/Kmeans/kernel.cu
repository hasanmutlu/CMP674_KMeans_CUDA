#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iostream>

#define int_ptr int*
#define float_ptr float*
#define X(point) point[0] // returns X of given point
#define Y(point) point[1] // returns Y of given point
#define POINT(list,i) &(list[i]) //returns i. point from given list
#define SIZE(count) count * 2

using namespace std;
using namespace chrono;

//Get distance of point to given Centroid
__device__ float get_distance_to_centroid(float_ptr centroid, float_ptr point)
{
	float diffX = X(point) - X(centroid);
	float diffY = Y(point) - Y(centroid);
	return sqrtf(diffX * diffX + diffY * diffY);
}

//Find centroid which distance is minumum to given point 
__device__ int get_closest_centroid(float_ptr centroids , float_ptr point, int centroid_count)
{
	int min = 0;
	float min_value = get_distance_to_centroid(POINT(centroids, 0), point);
	for (int i=2 ; i< SIZE(centroid_count); i+=2)
	{
		float distance = get_distance_to_centroid(POINT(centroids,i), point);
		if (distance < min_value)
		{
			min = i / 2;
			min_value = distance;
		}
	}
	return min;
}

//recenter given centroid according to points of it.
__global__ void recenter_centroids(float_ptr points, int point_count,int_ptr centroidPointMap, float_ptr centroids, int_ptr centroidPointCount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < point_count)
	{
		float_ptr point = POINT(points, id*2);
		float_ptr centroid = POINT(centroids, centroidPointMap[id] * 2);
		int pointCount = centroidPointCount[centroidPointMap[id]];
		atomicAdd(&(X(centroid)), (X(point) / pointCount));
		atomicAdd(&(Y(centroid)), (Y(point) / pointCount));
	}
}

__global__ void kmeans(float_ptr points, int point_count, float_ptr centroids, int centroidCount, int_ptr centroidPointMap, int_ptr centroidPointCount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < point_count)
	{
		float_ptr point = POINT(points, id * 2);
		int closest_centroid = get_closest_centroid(centroids, point, centroidCount);
		centroidPointMap[id] = closest_centroid;
		atomicAdd(&(centroidPointCount[closest_centroid]), 1);
	}
}


int getBlockCount(int count)
{
	int blockCount = 0;
	if (count % 32 == 0)
	{
		blockCount = count / 32;
	}
	else
	{
		blockCount = ((int)(count / 32)) + 1;
	}
	return blockCount;
}

void GenerateRandomPoint(int count, float_ptr &result)
{
	result = new float[SIZE(count)];
	for (int i = 0; i < SIZE(count); i+=2)
	{
		float randX = static_cast<float>((rand() % 100 * 2) - 100);
		float randY = static_cast<float>((rand() % 100 * 2) - 100);
		result[i] = randX;
		result[i+1] = randY;
	}
}


void print_points(float_ptr list, int count) 
{
	for (int i=0;i<count * 2;i+=2)
	{
		float_ptr point = POINT(list, i);
		float x = X(point);
		float y = Y(point);
		printf("%d -> (%lf,%lf)\n",i/2,x,y);
	}
}

bool isCentroidChanged(float_ptr curCentroid, float_ptr prevCentroid)
{
	int cX =static_cast<int>(X(curCentroid) * 100);
	int cY =static_cast<int>(Y(curCentroid) * 100);
	int pX =static_cast<int>(X(prevCentroid) * 100);
	int pY =static_cast<int>(Y(prevCentroid) * 100);
	return (cX != pX) || (cY != pY);
}

bool isProcessCompleted(float_ptr cur_centroids, float_ptr prev_centroids, int count)
{
	bool result = true;
	for (int i=0;i<SIZE(count);i+=2)
	{
		float_ptr c_centroid = POINT(cur_centroids, i);
		float_ptr p_centroid = POINT(prev_centroids, i);
		if (isCentroidChanged(c_centroid, p_centroid) == true)
		{
			result = false;
			break;
		}
	}
	return result;
}


void cudaKmeansProcess(float_ptr devPoints, float_ptr devCentroids, int_ptr devPointCount, int_ptr devPointMap,float_ptr centroids, int pointCount, int centroidCount, int blockCount)
{
	cudaMemset(devPointCount, 0, centroidCount * sizeof(float));
	kmeans <<<blockCount, 32 >>>(devPoints, pointCount, devCentroids, centroidCount, devPointMap, devPointCount);
	cudaDeviceSynchronize();
	cudaMemset(devCentroids, 0, SIZE(centroidCount) * sizeof(float));
	recenter_centroids <<<blockCount, 32 >> > (devPoints, pointCount, devPointMap, devCentroids, devPointCount);
	cudaMemcpy(centroids, devCentroids, SIZE(centroidCount) * sizeof(float), cudaMemcpyDeviceToHost);
}

void PrintToFile(const char *file, float_ptr points, float_ptr centroids,int pointCount,int centroidCount,int_ptr centroidPointMap,int_ptr centroidPointCount)
{
	fstream output(file, fstream::out);
	for (int i=0;i<centroidCount;i++)
	{
		int cid = i * 2;
		float_ptr centroid = POINT(centroids,cid);
		int count = centroidPointCount[i];
		output << X(centroid) << "," << Y(centroid) << "," << count << endl;
		for (int j=0 ; j<pointCount;j++)
		{
			if (centroidPointMap[j] == i)
			{
				float_ptr point = POINT(points, j*2);
				output << X(point) << "," << Y(point) << endl;
			}
		}
	}
}

void cudaKmeans(int pointCount, int centroidCount)
{
	float_ptr points, *devPoints;
	float_ptr centroids, *devCentroids;
	float_ptr prev_centroids = new float[SIZE(centroidCount)];
	int_ptr centroidPointMap = reinterpret_cast<int_ptr>(calloc(pointCount, sizeof(int)));
	int_ptr centroidPointCount = reinterpret_cast<int_ptr>(calloc(centroidCount, sizeof(int)));
	int_ptr devPointMap;
	int_ptr devPointCount;
	auto start = system_clock::now();
	GenerateRandomPoint(pointCount, points);
	GenerateRandomPoint(centroidCount, centroids);
	int blockCount = getBlockCount(pointCount);
	cudaMalloc((void **)&devPoints, SIZE(pointCount) * sizeof(float));
	cudaMalloc((void **)&devCentroids, SIZE(centroidCount) * sizeof(float));
	cudaMalloc((void **)&devPointMap, pointCount * sizeof(int));
	cudaMalloc((void **)&devPointCount, centroidCount * sizeof(int));
	cudaMemcpy(devPoints, points, SIZE(pointCount) * sizeof(float), cudaMemcpyHostToDevice);
	bool found = false;
	while (found == false)
	{
		memcpy(prev_centroids, centroids, sizeof(float)*SIZE(centroidCount));
		cudaMemcpy(devCentroids, centroids, SIZE(centroidCount) * sizeof(float), cudaMemcpyHostToDevice);
		cudaKmeansProcess(devPoints, devCentroids, devPointCount, devPointMap, centroids, pointCount, centroidCount, blockCount);
		found = isProcessCompleted(centroids, prev_centroids, centroidCount);
	}
	auto end = system_clock::now();
	auto elapsed_seconds = duration_cast<milliseconds>(end - start).count();
	cout << "Elapsed Time:" << elapsed_seconds << "ms" << endl;
	cudaMemcpy(centroids, devCentroids, SIZE(centroidCount) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(centroidPointCount, devPointCount, centroidCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(centroidPointMap, devPointMap, pointCount * sizeof(int), cudaMemcpyDeviceToHost);
	PrintToFile("cuda_output.txt",points, prev_centroids,pointCount, centroidCount,centroidPointMap,centroidPointCount);
	cudaFree(devPoints);
	cudaFree(devCentroids);
	cudaFree(devPointMap);
	cudaFree(devPointCount);

}




int main(int argc, char *argv[])
{
	int centroidCount = 0;
	int pointCount = 0;
	if (argc > 2)
	{
		srand(time(NULL));
		pointCount = atoi(argv[1]);
		centroidCount = atoi(argv[2]);
		cudaDeviceReset();
		cudaKmeans(pointCount, centroidCount);
		cudaDeviceReset();
	}
	return 1;
}


