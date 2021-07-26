#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <vector>
#include <type_traits>
#include <iterator>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cassert>

// #include <thread>
// #include <future>
#include <time.h>

#define TYPE_VAR float

class NFOV
{
public:
    NFOV(int height, int width);
    void set_screen_points(int height, int width);
    void calcSphericaltoGnomonic(cv::Point_<TYPE_VAR> center_point);
    void bilinear_interpolation(cv::Mat &frame, cv::Mat &result);

private:
    int height, width;
    cv::Point_<TYPE_VAR> FOV = cv::Point_<TYPE_VAR>(0.45, 0.45);
    TYPE_VAR sin_cp_y, cos_cp_y;
    cv::Point_<TYPE_VAR> cp;
    cv::Mat_<TYPE_VAR> xx, yy, x, y, rou;
    cv::Mat_<TYPE_VAR> c, sin_c, cos_c;
    cv::Mat_<TYPE_VAR> Lat, Lon;
};


