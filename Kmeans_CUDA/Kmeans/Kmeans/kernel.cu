#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>


__global__ void setup_rand_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_rand_kernel(curandState *state, int n,	int *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int x;
	curandState localState = state[id];
	x = (curand(&localState) % 100 * 2) - 100;
	state[id] = localState;
	result[id] += x;
}

__device__ double get_distance_to_centroid(double *centroid, int *point)
{
	int point_x = point[0];
	int point_y = point[1];
	double centroid_x = centroid[0];
	double centroid_y = centroid[1];
	double diffX = point_x - centroid_x;
	double diffY = point_y - centroid_y;
	return sqrtf(diffX * diffX + diffY * diffY);
}

__device__ int get_min_distance_centroid(double *centroids , int *point, int centroid_count)
{
	int point_x = point[0];
	int point_y = point[1];
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

__device__ double *recenter_centroid(double *centroid, int *points, int point_count)
{
	double result[2];
	result[0] = 0;
	result[1] = 0;
	for (int i=0 ; i<point_count *2 ; i+=2)
	{
		result[0] += points[i] / point_count;
		result[1] += points[i + 1] / point_count;
	}
	return result;
}

__global__ void kmeans(int *points, int *result )
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

int *kmeans_cuda()
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


int main(int argc, char *argv[])
{
	int sampleCount = 5 * 2;
	int *result = kmeans_cuda();
	for (int i = 0; i < sampleCount; i += 2) {
		int &x = result[i];
		int &y = result[i + 1];
		printf("X: %d , Y: %d\n", x, y);
	}
	system("pause");
	return 1;
}


