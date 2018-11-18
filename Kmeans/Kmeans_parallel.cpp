#include <iostream>
#include <sstream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <thread>
#include <mutex>

#define uint unsigned int
#define RANGE 100

using namespace std;

mutex mutex_obj1,mutex_obj2;
class Point
{
public:
	int x, y;
	int operator[](int index)
	{
		if (index <= 0)
		{
			return x;
		}
		else
		{
			return y;
		}
	}
	Point(int x, int y)
	{
		this->x = x;
		this->y = y;
	}
	Point():Point(0, 0){}
	string ToString() const
	{
		stringstream ss;// = new stringstream();
        //ss << "[" << x << "," << y << "]" << endl;
        ss <<  x << "," << y;
		return ss.str();
	}
	double GetDistance(Point point)
	{
		double diffX = x - point.x;
		double diffY = y - point.y;
		return sqrt(diffX * diffX + diffY * diffY);
	}
};

class Centroid
{
public:
	vector<Point> points;
	Point position;
	Point operator[](int index) {
		return points[index];
	}
	Centroid(int x, int y)
	{
		position.x = x;
		position.y = y;
	}
	Centroid(Point p)
	{
		position = p;
	}
	string ToString()
	{
		stringstream ss;
		ss << "[" << position.x << "," << position.y << "]";
		return ss.str();
	}
	void Print()
	{
		cout << "Centroid " << ToString() << endl;
		cout << "*********************" << endl;
		for (int i=0; i<points.size();i++)
		{
			cout << points[i].ToString() << endl;
		}

	}
	double GetDistance(Point point)
	{
		return position.GetDistance(point);
	}
	void UpdateCenter()
	{
		int count = points.size();
		if (count > 0)
		{
			Point sum(0, 0);
			for (uint i = 0; i < count; i++)
			{
				sum.x += points[i].x;
				sum.y += points[i].y;
			}
			sum.x = sum.x / count;
			sum.y = sum.y / count;
			position = sum;
		}
	}
};
Point GenerateRandomPoint()
{
	int randX = ( rand() % RANGE * 2 ) - RANGE;
	int randY = ( rand() % RANGE * 2 ) - RANGE;
	return Point(randX, randY);
}
vector<Point> GenerateRandomPoints(int size)
{
	vector<Point> *result = new vector<Point>();
	for (uint i = 0; i<size; i++)
	{
		Point p = GenerateRandomPoint();
		result->push_back(p);
	}
	return *result;
}
void ClearCentroids(vector<Centroid> &centroids)
{
	for (uint i=0; i < centroids.size(); i++)
	{
		centroids[i].points.clear();
	}
}

void UpdateCentroid(Centroid &centroid)
{
	centroid.UpdateCenter();
}

void UpdateCentroids(vector<Centroid> &centroids)
{
	vector<thread> thread_list;
	for (uint i = 0; i < centroids.size(); i++)
	{
		thread_list.push_back(thread(UpdateCentroid,ref(centroids[i])));
	}
	for (uint i = 0; i < centroids.size(); i++)
	{
		thread_list[i].join();
	}
}
bool IsCentroidsUpdated(const vector<Centroid> &previous, const vector<Centroid> &current)
{
	for (uint i = 0; i<previous.size(); i++)
	{
		Point prev = previous[i].position;
		Point cur = current[i].position;
		if (prev.x != cur.x || prev.y != cur.y)
		{
			return true;
		}
	}
	return false;
}

void FindClosestCentroid(const Point &p, vector<Centroid> &centroids)
{
	int minCentroidIndex = 0;
	for (uint j = 0; j<centroids.size(); j++)
	{
		double cur_distance = centroids[j].GetDistance(p);
		double min_distance = centroids[minCentroidIndex].GetDistance(p);
		if (cur_distance <= min_distance)
		{
			minCentroidIndex = j;
		}
	}
	centroids[minCentroidIndex].points.push_back(p);
}



void CalculateCentroids(vector<Centroid> &centroids, const vector<Point> &points)
{
	vector<Centroid> previousCentroids = centroids;
	//vector<thread> thread_list;		
	uint point_size = points.size();
	for (uint i = 0; i< point_size; i++)
	{
		const Point &p = points[i];
		FindClosestCentroid(p,centroids);
		//thread_list.push_back(thread(FindClosestCentroid,ref(p),ref(centroids)));
	 }
	// for (uint i = 0; i< point_size; i++)
	// {
	// 	thread_list[i].join();
	// }
	UpdateCentroids(centroids);
	bool isUpdated = IsCentroidsUpdated(previousCentroids, centroids);
	if (isUpdated == true)
	{
		ClearCentroids(centroids);
		CalculateCentroids(centroids, points);
	}
}

vector<Centroid> Kmeans(const vector<Point> &points, int k)
{
	vector<Centroid> centroids;
	for (int i = 0; i<k; i++)
	{
		Point p = GenerateRandomPoint();
		Centroid c(p);
		centroids.push_back(c);
	}
	CalculateCentroids(centroids, points);
	return centroids;
}
void PrintToFile(const char *file, const vector<Centroid> &centroids )
{
    fstream output(file, fstream::out);
    uint c_size = centroids.size();
    for (int i=0; i<c_size; i++){
        const Centroid &centroid = centroids[i];
        uint point_size = centroid.points.size();
        output << centroid.position.ToString()<<","<< point_size<< endl;
        for (uint j= 0; j< point_size ; j++)
        {
            const Point &p = centroid.points[j];            
            output<<p.ToString()<<endl;
        }
    }
    output.close();
}

int main(int argc, char *args[])
{
	srand(time(NULL));
    uint data_count = 10;
    uint cluster_count = 3;
    if (argc > 2)
    {
        data_count = atoi(args[1]);
        cluster_count = atoi(args[2]);
    }
	auto data = GenerateRandomPoints(data_count);
	clock_t start = clock();
	auto result = Kmeans(data, cluster_count);
	clock_t end = clock();
	double elapsed = (end - start)/1000.0;
    //PrintToFile("output_parallel.txt",result);
	// for (uint i=0;i<result.size();i++)
	// {
	// 	result[i].Print();
	// }
	cout<<"Parallel Elapsed Time:"<<elapsed<<"ms"<<endl;
	return -1;
}
