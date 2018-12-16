#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define int_ptr int*
#define float_ptr float*
#define list_ptr float**
#define X(point) point[0] // returns X of given point
#define Y(point) point[1] // returns Y of given point
#define POINT(list,i) &(list[i]) //returns i. point from given list
#define SIZE(count) count * 2

#pragma region Random initializer methods
__global__ void setup_rand_kernel(curandState *state, int sampleCount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < sampleCount)
	{
		curand_init(sampleCount, id, 0, &state[id]);
	}
}

__global__ void generate_rand_kernel(curandState *state, int sampleCount, float_ptr result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < sampleCount)
	{
		curandState localState = state[id];
		int x = (curand(&localState) % 100 * 2) - 100;
		state[id] = localState;
		result[id] += x;
	}
}

#pragma endregion

//Get distance of point to given Centroid
__device__ double get_distance_to_centroid(float_ptr centroid, int_ptr point)
{
	int point_x = X(point);
	int point_y = Y(point);
	float centroid_x = X(centroid);
	float centroid_y = Y(centroid);
	float diffX = point_x - centroid_x;
	float diffY = point_y - centroid_y;
	return sqrtf(diffX * diffX + diffY * diffY);
}

//Find centroid which distance is minumum to given point 
__device__ int get_min_distance_centroid(float_ptr centroids , int_ptr point, int centroid_count)
{
	int point_x = X(point);
	int point_y = Y(point);
	int min = 0;
	float min_value = 0;
	for (int i=0 ; i< centroid_count; i+=2)
	{
		float distance = get_distance_to_centroid(centroids + i, point);
		if (min == 0 || distance < min_value)
		{
			min = i / 2;
			min_value = distance;
		}
	}
	return min;
}

//recenter given centroid according to points of it
__device__ float_ptr recenter_centroid(float_ptr centroid, int_ptr points, int point_count)
{
	float result[2];
	X(result) = 0;
	Y(result) = 0;
	for (int i=0 ; i<point_count *2 ; i+=2)
	{
		X(result) += points[i] / point_count;
		Y(result) += points[i + 1] / point_count;
	}
	return result;
}

__global__ void kmeans(int_ptr points, int_ptr result )
{
	int point_count = 4096;
	int centroid_count = 5;
	for (int i=0;i<point_count *2 ; i+=2)
	{



	}
}


//Birinci Algoritma
//1.random noktalari olustur OK
//2.random centroid merkezleri olustur OK
//3.geriye her bir centroid icin olusturulan sayi kadar olacak sekilde centroidlerin noktalarini bul ve geriye dondur
//4.her bir centroid in suanki merkez noktalarini tut
//5.merkez noktalarini guncelle
//6.eger merkez degismisse 3. adima git 
//7.ciktilari ekrana ve dosyaya yazdir
//8.ciktilari python kodu ile ekrana cizdirebilirsin

void cudaGetRandomPoints(int count, float_ptr &result )
{
	curandState *devStates;
	float_ptr devResults;
	int sampleCount = count * 2;
	int blockCount = 0;
	if (count % 32 == 0)
	{
		blockCount = sampleCount / 32;
	}
	else
	{
		blockCount = ((int)(sampleCount / 32)) + 1;
	}
	result = new float[sampleCount];
	cudaMalloc((void **)&devResults, sampleCount * sizeof(float));
	cudaMemset(devResults, 0, sampleCount * sizeof(float));
	cudaMalloc((void **)&devStates, sampleCount * sizeof(curandState));
	setup_rand_kernel << <blockCount, 32 >> >(devStates, sampleCount);
	generate_rand_kernel << <blockCount, 32 >> >(devStates, sampleCount, devResults);
	cudaMemcpy(result, devResults, sampleCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(devStates);
	cudaFree(devResults);
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
	return X(curCentroid) != X(prevCentroid) || Y(curCentroid) != Y(prevCentroid);
}

bool checkProcessCompleted(float_ptr cur_centroids, float_ptr prev_centroids, int count)
{
	bool result = true;
	for (int i=0;i<SIZE(count);i+=2)
	{
		float_ptr c_centroid = POINT(cur_centroids, i);
		float_ptr p_centroid = POINT(prev_centroids, i);
		if (isCentroidChanged(c_centroid, p_centroid))
		{

		}


	}
	return result;

}

void createCentroidsPoints(int centroidCount, int pointCount, list_ptr &result)
{
	result = reinterpret_cast<float_ptr*>(calloc(centroidCount, sizeof(float_ptr)));
	for (int i=0;i<centroidCount;i++)
	{
		result[i] = reinterpret_cast<float_ptr>(calloc(SIZE(pointCount),sizeof(float)));
	}
}


void cudaKmeans()
{
	cudaDeviceReset();
	int centroidCount = 5;
	int pointCount = 4096;
	float_ptr points;
	float_ptr centroids;
	list_ptr centroidPoints;
	int_ptr pointCountOfCentroids = new int[centroidCount];
	for (int i = 0; i<centroidCount;i++)
	{
		pointCountOfCentroids[i] = pointCount;
	}
	cudaGetRandomPoints(pointCount, points);
	cudaGetRandomPoints(centroidCount, centroids);
	createCentroidsPoints(centroidCount, pointCount, centroidPoints);
	for (int i=0;i<centroidCount;i++)
	{
		print_points(centroidPoints[i],pointCount);
		system("pause");

	}
	bool found = false;
	while (found == false)
	{

		//cudaKmeansProcess(points, pointCount, centroids, centroidCount);
		//cudaRecenterCentroids()
		//found = is
		found = true;
	}



	/*
	hostResult = new int[centroidCount * 2];
	
	cudaMalloc((void**)devPoints, pointCount * 2 * sizeof(int));
	cudaMemcpy(devPoints, points, pointCount * 2 * sizeof(int),cudaMemcpyHostToDevice);

	cudaMalloc((void**)devResult, centroidCount * 2 * sizeof(int));
	cudaMemset(devResult, 0 , centroidCount * 2 * sizeof(int));

	//kmeans << <64, 64 >> > (devPoints, devResult);
	cudaMemcpy(hostResult, devResult, centroidCount * 2 * sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(devPoints);
	cudaFree(devResult);
	cudaDeviceReset();
	*/



}


int main(int argc, char *argv[])
{
	cudaKmeans();
	system("pause");
	return 1;
}


