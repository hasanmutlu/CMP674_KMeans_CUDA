#include <iostream>
#include <sstream>
#include <math.h>
#include <vector>
#include <cstdlib>

using namespace std;
class Point
{
    public:
        int x,y;
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
        Point():Point(0,0)
        {
        }
        string ToString()
        {
            stringstream ss;// = new stringstream();
            ss<<"["<<x<<","<<y<<"]"<<endl;
            return ss.str();
        }
        float GetDistance(Point point)
        {
            double diffX = x - point.x;
            double diffY = y - point.y;
            return sqrt( diffX * diffX + diffY * diffY );
        }
};

class Centroid
{
    public:
        vector<Point> points;
        Point position;
        Point operator[](int index){
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
            stringstream ss;// = new stringstream();
            ss<<"["<<position.x<<","<<position.y<<"]";
            return ss.str();
        }
        void Print()
        {
            cout<<"Centroid "<<ToString()<<endl;
            cout<<"*********************"<<endl;
            for (auto pointIt = points.begin(); pointIt!= points.end();++pointIt)
            {
                cout<<pointIt->ToString()<<endl;
            }

        }
        float GetDistance(Point point)
        {
            return position.GetDistance(point);
        }
        void UpdateCenter()
        {
            if (points.size()> 0)
            {
                Point sum(0,0);
                for(auto point = points.begin() ; point!= points.end();++point)
                {
                    sum.x += point->x;
                    sum.y += point->y;
                }
                int count = points.size();
                sum.x = sum.x / count;
                sum.y = sum.y / count;
                position = sum;
            }

        }
};
Point GenerateRandomPoint()
{
    int randX = rand() % 200 - 100;
    int randY = rand() % 200 - 100;
    return Point(randX, randY);
}
vector<Point> GenerateRandomPoints(int size)
{
    vector<Point> *result = new vector<Point>();
    for (int i=0 ; i<size ; i++)
    {
        Point p = GenerateRandomPoint();
        result->push_back(p);
    }
    return *result;
}
void ClearCentroids(vector<Centroid> &centroids)
{
    for (auto iterator = centroids.begin(); iterator != centroids.end();++iterator)
    {
        Centroid c = *iterator;
        c.points.clear();
    }
}
void UpdateCentroids(vector<Centroid> &centroids)
{
    for (auto iterator = centroids.begin(); iterator != centroids.end();++iterator)
    {
        Centroid &c = *iterator;
        c.UpdateCenter();
    }
}
bool IsCentroidsUpdated(const vector<Centroid> &previous, const vector<Centroid> &current)
{
    for (int i=0; i<previous.size();i++)
    {
        Point prev = previous[i].position;
        Point cur = current[i].position;
        if ( prev.x != cur.x || prev.y != cur.y )
        {
            return true;
        }

    }
    return false;
}

void CalculateCentroids(vector<Centroid> &centroids,const vector<Point> &points)
{
    vector<Centroid> previousCentroids = centroids;
    for (auto pointIt = points.begin() ; pointIt!=points.end();++pointIt)
    {
        Point p = *pointIt;
        int minCentroidIndex = 0;
        for (int i=0; i<centroids.size();i++)
        {
            float cur_distance = centroids[i].GetDistance(p);
            float min_distance = centroids[minCentroidIndex].GetDistance(p);
            if (cur_distance <= min_distance)
            {
                minCentroidIndex = i;
            }
        }
        centroids[minCentroidIndex].points.push_back(p);
    }
    UpdateCentroids(centroids);
    bool isUpdated = IsCentroidsUpdated(previousCentroids, centroids);
    if (isUpdated == true)
    {
        ClearCentroids(centroids);
        CalculateCentroids(centroids,points);
    }




}

vector<Centroid> Kmeans(const vector<Point> &points, int k )
{
    vector<Centroid> centroids;
    for (int i=0 ; i<k; i++)
    {
        Point p = GenerateRandomPoint();
        Centroid c(p);
        centroids.push_back(c);
    }
    CalculateCentroids(centroids, points);
    return centroids;
}

int main(int argc, char *args[])
{
    srand(time(NULL));
    /*
    Point a(0,0);
    Centroid c(a);
    Point b(3 , 4);
    double distance = c.GetDistance(b);
    cout<<distance<<endl;
    Point *data = GenerateRandomPoints(10); 
    for (int i=0 ; i<10 ; i++)
    {
        cout<<data[i].ToString()<<endl;

    }
    for(auto i = data.begin();i!=data.end();++i)
    {
        cout<<i->ToString()<<endl;
    }
    */
    auto points = GenerateRandomPoints(10); 
    auto result = Kmeans(points, 3);
    for(auto i = result.begin();i!=result.end();++i)
    {
        cout<<i->points.size()<<endl;
        //i->Print();
    }


    return -1;
}
