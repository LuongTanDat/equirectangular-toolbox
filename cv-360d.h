#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

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

template <typename T>
T mod1(T i)
{
    return static_cast<T>(i < 0 ? 1 + i : i);
}

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

namespace np
{
    template <typename it, typename T>
    void linspace(it start_in, it end_in, int num_in, std::vector<T> &out);

    template <typename T>
    void meshgrid(std::vector<T> &xgv, std::vector<T> &ygv, std::vector<T> &X, std::vector<T> &Y);

    template <typename T>
    void meshgrid(cv::Mat_<T> &xgv, cv::Mat_<T> &ygv, cv::Mat_<T> &X, cv::Mat_<T> &Y);

    template <typename T>
    void op3(cv::Mat_<T> &result, cv::Mat_<T> &X, TYPE_VAR (*func)(TYPE_VAR));

    template <typename T>
    void sin(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void cos(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void atan(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void asin(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void atan2(cv::Mat_<T> &result, cv::Mat_<T> &Y, cv::Mat_<T> &X);

    template <typename T>
    void mod_1(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void floor(cv::Mat_<T> &result, cv::Mat_<T> &X);

    template <typename T>
    void take(cv::Mat_<cv::Vec3f> &result, cv::Mat &img, cv::Mat_<TYPE_VAR> &Idx, cv::Mat_<TYPE_VAR> &W);
}

template <typename it, typename T>
void np::linspace(it start_in, it end_in, int num_in, std::vector<T> &out)
{
    out.clear();

    T start = static_cast<T>(start_in);
    T end = static_cast<T>(end_in);
    T num = static_cast<T>(num_in);

    if (num == 0)
    {
    }
    if (num == 1)
    {
        out.push_back(start);
    }

    T delta = (end - start) / (num - 1);
    for (int i = 0; i < num - 1; ++i)
    {
        out.push_back(start + delta * i);
    }
    out.push_back(end); // I want to ensure that start and end
                        // are exactly the same as the input
}

template <typename T>
void np::meshgrid(cv::Mat_<T> &xgv, cv::Mat_<T> &ygv, cv::Mat_<T> &X, cv::Mat_<T> &Y)
{
    cv::repeat(xgv, ygv.total(), 1, X);
    X = X.reshape(1, 1);
    cv::repeat(ygv.t(), 1, xgv.total(), Y);
    Y = Y.reshape(1, 1);
}

template <typename T>
void np::meshgrid(std::vector<T> &xgv, std::vector<T> &ygv, std::vector<T> &X, std::vector<T> &Y)
{
    cv::Mat_<TYPE_VAR> _xgv = cv::Mat_<TYPE_VAR>(xgv).reshape(1, 1);
    cv::Mat_<TYPE_VAR> _ygv = cv::Mat_<TYPE_VAR>(ygv).reshape(1, 1);
    cv::Mat_<TYPE_VAR> X_mat1f, Y_mat1f;
    cv::repeat(_xgv, _ygv.total(), 1, X_mat1f);
    X_mat1f = X_mat1f.reshape(1, 1);
    cv::repeat(_ygv.t(), 1, _xgv.total(), Y_mat1f);
    Y_mat1f = Y_mat1f.reshape(1, 1);
    cv::Mat_<TYPE_VAR>::iterator x_iter = X_mat1f.begin();
    cv::Mat_<TYPE_VAR>::iterator y_iter = Y_mat1f.begin();
    X.clear();
    Y.clear();
    while ((x_iter != X_mat1f.end()) && (y_iter != Y_mat1f.end()))
    {
        X.push_back(*x_iter);
        Y.push_back(*y_iter);
        x_iter++;
        y_iter++;
    }
}

template <typename T>
void np::op3(cv::Mat_<T> &result, cv::Mat_<T> &X, TYPE_VAR (*func)(TYPE_VAR))
{
    std::vector<T> _X;
    for (typename cv::Mat_<T>::iterator iter = X.begin(); iter != X.end(); iter++)
    {
        _X.push_back(static_cast<T>(func(*iter)));
    }
    cv::Mat_<T>(_X).reshape(1, 1).copyTo(result);
}

template <typename T>
void np::sin(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &std::sin);
}

template <typename T>
void np::cos(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &std::cos);
}

template <typename T>
void np::atan(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &std::atan);
}

template <typename T>
void np::asin(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &std::asin);
}

template <typename T>
void np::atan2(cv::Mat_<T> &result, cv::Mat_<T> &Y, cv::Mat_<T> &X)
{
    std::vector<T> _ATAN2;
    typename cv::Mat_<T>::iterator x_iter = X.begin();
    typename cv::Mat_<T>::iterator y_iter = Y.begin();
    while ((x_iter != X.end()) && (y_iter != Y.end()))
    {
        _ATAN2.push_back(static_cast<T>(std::atan2(*y_iter, *x_iter)));
        x_iter++;
        y_iter++;
    }
    cv::Mat_<T>(_ATAN2).reshape(1, 1).copyTo(result);
}

template <typename T>
void np::mod_1(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &mod1);
}

template <typename T>
void np::floor(cv::Mat_<T> &result, cv::Mat_<T> &X)
{
    np::op3<T>(result, X, &std::floor);
}

template <typename T>
void np::take(cv::Mat_<cv::Vec3f> &result, cv::Mat &img, cv::Mat_<TYPE_VAR> &Idx, cv::Mat_<TYPE_VAR> &W)
{
    int cnt = 0;
    cv::Mat temp;
    // img.convertTo(temp, CV_32FC3);
    typename cv::Mat_<TYPE_VAR>::iterator idx_iter = Idx.begin();
    typename cv::Mat_<TYPE_VAR>::iterator w_iter = W.begin();
    while (idx_iter != Idx.end() && w_iter != W.end())
    {
        // std::cout << cnt << "\t" << *iter << "\t" << result.at<cv::Vec3f>(0, cnt) << "\t" << temp.at<cv::Vec3f>(0, *iter) << std::endl;
        result.at<cv::Vec3f>(0, cnt++) = img.at<cv::Vec3b>(0, static_cast<T>(*idx_iter)) * *w_iter;
        idx_iter++;
        w_iter++;
    }
}
