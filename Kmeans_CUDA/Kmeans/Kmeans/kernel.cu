#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define int_ptr int*
#define float_ptr float*
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
		result[id] = x;
	}
}

#pragma endregion

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
		atomicAdd(&(X(centroid)), (X(point) / pointCount));//TODO: will be optimized
		atomicAdd(&(Y(centroid)), (Y(point) / pointCount));//TODO: will be optimized
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
		atomicAdd(&(centroidPointCount[closest_centroid]), 1);//TODO: will be optimized
		//centroidPointCount[closest_centroid]++;
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


void cudaGetRandomPoints(int count, float_ptr &result )
{
	curandState *devStates;
	float_ptr devResults;
	int sampleCount = count * 2;
	int blockCount = getBlockCount(sampleCount);
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


void cudaKmeansProcess(float_ptr points, int pointCount, float_ptr centroids, int centroidCount, int_ptr centroidPointMap, int_ptr centroidPointCount)
{
	float_ptr devPoints;
	float_ptr devCentroids;
	float_ptr devNewCentroids;
	int_ptr devPointMap;
	int_ptr devPointCount;
	int blockCount = getBlockCount(pointCount);
	cudaMalloc((void **)&devPointMap, pointCount * sizeof(int));
	cudaMemset(devPointMap, 0, pointCount * sizeof(int));
	cudaMalloc((void **)&devPointCount, centroidCount * sizeof(int));
	cudaMemset(devPointCount, 0, centroidCount * sizeof(int));
	cudaMalloc((void **)&devPoints, SIZE(pointCount) * sizeof(float));
	cudaMemset(devPoints, 0, SIZE(pointCount) * sizeof(float));
	cudaMalloc((void **)&devCentroids, SIZE(centroidCount) * sizeof(float));
	cudaMemset(devCentroids, 0, SIZE(centroidCount) * sizeof(float));
	cudaMalloc((void **)&devNewCentroids, SIZE(centroidCount) * sizeof(float));
	cudaMemset(devNewCentroids, 0, SIZE(centroidCount) * sizeof(float));
	memset(centroidPointMap, 0, pointCount * sizeof(int));
	memset(centroidPointCount, 0, centroidCount * sizeof(int));
	cudaMemcpy(devPoints, points, SIZE(pointCount) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCentroids, centroids, SIZE(centroidCount) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devPointMap, centroidPointMap, pointCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devPointCount, centroidPointCount, centroidCount * sizeof(int), cudaMemcpyHostToDevice);
	kmeans<<<blockCount,32>>>(devPoints, pointCount, devCentroids, centroidCount, devPointMap, devPointCount);
	cudaDeviceSynchronize();
	cudaMemset(devCentroids, 0, SIZE(centroidCount) * sizeof(float));
	recenter_centroids <<<blockCount, 32 >>> (devPoints, pointCount, devPointMap, devNewCentroids, devPointCount);
	cudaMemcpy(centroidPointMap, devPointMap, pointCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(centroidPointCount, devPointCount, centroidCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(centroids, devNewCentroids, SIZE(centroidCount) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(devPoints);
	cudaFree(devCentroids);
	cudaFree(devNewCentroids);
	cudaFree(devPointMap);
	cudaFree(devPointCount);
}


int main(int argc, char *argv[])
{
	cudaDeviceReset();
	int centroidCount = 3;
	int pointCount = 10;
	float_ptr points;
	float_ptr centroids;
	float_ptr prev_centroids = new float[SIZE(centroidCount)];
	int_ptr centroidPointMap = reinterpret_cast<int_ptr>(calloc(pointCount, sizeof(int)));
	int_ptr centroidPointCount = reinterpret_cast<int_ptr>(calloc(centroidCount, sizeof(int)));
	cudaGetRandomPoints(pointCount, points);
	cudaGetRandomPoints(centroidCount, centroids);
	bool found = false;
	while (found == false)
	{
		memcpy(prev_centroids, centroids, sizeof(float)*SIZE(centroidCount));
		cudaKmeansProcess(points, pointCount, centroids, centroidCount,centroidPointMap,centroidPointCount);
		found = isProcessCompleted(centroids, prev_centroids, centroidCount);
	}
	print_points(centroids, centroidCount);
	cudaDeviceReset();
	system("pause");
	return 1;
}


