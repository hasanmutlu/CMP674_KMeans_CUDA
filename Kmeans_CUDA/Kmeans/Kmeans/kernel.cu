#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define int_ptr int*
#define double_ptr double*
#define X(point) point[0]
#define Y(point) point[1]

#pragma region Random initializer methods
__global__ void setup_rand_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_rand_kernel(curandState *state, int n, int_ptr result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int x;
	curandState localState = state[id];
	x = (curand(&localState) % 100 * 2) - 100;
	state[id] = localState;
	result[id] += x;
}

#pragma endregion

//Get distance of point to given Centroid
__device__ double get_distance_to_centroid(double_ptr centroid, int_ptr point)
{
	int point_x = X(point);
	int point_y = Y(point);
	double centroid_x = X(centroid);
	double centroid_y = Y(centroid);
	double diffX = point_x - centroid_x;
	double diffY = point_y - centroid_y;
	return sqrtf(diffX * diffX + diffY * diffY);
}

//Find centroid which distance is minumum to given point 
__device__ int get_min_distance_centroid(double_ptr centroids , int_ptr point, int centroid_count)
{
	int point_x = X(point);
	int point_y = Y(point);
	int min = 0;
	double min_value = 0;
	for (int i=0 ; i< centroid_count; i+=2)
	{
		double distance = get_distance_to_centroid(centroids + i, point);
		if (min == 0 || distance < min_value)
		{
			min = i / 2;
			min_value = distance;
		}
	}
	return min;
}

//recenter given centroid according to points of it
__device__ double_ptr recenter_centroid(double_ptr centroid, int_ptr points, int point_count)
{
	double result[2];
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


int *generate_random_points()
{ 
	curandState *devStates;
	int *devResults, *hostResults;
	int sampleCount = 4096 * 2;
	hostResults = new int[sampleCount];
	cudaMalloc((void **)&devResults, sampleCount * sizeof(int));
	cudaMemset(devResults, 0, sampleCount * sizeof(int));
	cudaMalloc((void **)&devStates, sampleCount * sizeof(curandState));
	setup_rand_kernel <<<64, 64 >>>(devStates);
	generate_rand_kernel <<<64, 64 >>>(devStates, sampleCount, devResults);
	cudaMemcpy(hostResults, devResults, sampleCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(devStates);
	cudaFree(devResults);
	return hostResults;

}

int_ptr kmeans_cuda()
{
	cudaDeviceReset();
	int *points, *devPoints, *devResult, *hostResult;
	int centroidCount = 5;
	int pointCount = 4096;
	points = generate_random_points();
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
	return hostResult;


}

//Birinci Algoritma
//1.random noktalari olustur
//2.random centroid merkezleri olustur
//3.geriye her bir centroid icin olusturulan sayi kadar olacak sekilde centroidlerin noktalarini bul ve geriye dondur
//4.her bir centroid in suanki merkez noktalarini tut
//5.merkez noktalarini guncelle
//6.eger merkez degismisse 3. adima git 
//7.ciktilari ekrana ve dosyaya yazdir
//8.ciktilari python kodu ile ekrana cizdirebilirsin


int main(int argc, char *argv[])
{
	int sampleCount = 5 * 2;
	int_ptr result = kmeans_cuda();
	for (int i = 0; i < sampleCount; i += 2) {
		int &x = X(result);
		int &y = Y(result);
		printf("X: %d , Y: %d\n", x, y);
	}
	system("pause");
	return 1;
}


