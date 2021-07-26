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
    std::vector<TYPE_VAR> xx, yy, x, y, rou;
    std::vector<TYPE_VAR> c, sin_c, cos_c;
    std::vector<TYPE_VAR> Lat, Lon;
};

namespace np
{
    template <typename it, typename T>
    void linspace(it start_in, it end_in, int num_in, std::vector<T> &out);

    template <typename T>
    void meshgrid(std::vector<T> &xgv, std::vector<T> &ygv, std::vector<T> &X, std::vector<T> &Y);

    template <typename T>
    void mul(std::vector<T> &result, std::vector<T> &X, T Y = 1, T Z = 0);

    template <typename T>
    void mul(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z = 0);

    template <typename T>
    void div(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z = 0);

    template <typename T>
    void add(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z = 0);

    template <typename T>
    void add(std::vector<T> &result, std::vector<T> &X, T Y, T Z = 0);

    template <typename T>
    void sub(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z = 0);

    template <typename T1, typename T2, typename T3>
    void sub(std::vector<T1> &result, std::vector<T2> &X, std::vector<T3> &Y);

    template <typename T>
    void pow(std::vector<T> &result, std::vector<T> &base, int exp);

    template <typename T>
    void sqrt(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void op2(std::vector<T> &result, std::vector<T> &X, TYPE_VAR (*func)(TYPE_VAR));

    template <typename T>
    void atan(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void atan2(std::vector<T> &result, std::vector<T> &Y, std::vector<T> &X);

    template <typename T>
    void asin(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void acos(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void op(std::vector<T> &result, std::vector<T> &X, TYPE_VAR (*func)(TYPE_VAR));

    template <typename T>
    void sin(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void cos(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void tan(std::vector<T> &result, std::vector<T> &X);

    template <typename T, typename T2>
    void cast(std::vector<T2> &result, std::vector<T> &X);

    template <typename T>
    void mod_1(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void take(std::vector<cv::Vec3f> &result, cv::Mat &img, std::vector<T> &Idx);

    template <typename T>
    void floor(std::vector<T> &result, std::vector<T> &X);

    template <typename T>
    void round(std::vector<T> &result, std::vector<T> &X);
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
void np::mul(std::vector<T> &result, std::vector<T> &X, T Y, T Z)
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
            *iter = *iter * Y + Z;
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
            result.push_back(*iter * Y + Z);
    }
}

template <typename T>
void np::mul(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z)
{
    assert(X.size() == Y.size());
    typename std::vector<T>::iterator x_iter = X.begin();
    typename std::vector<T>::iterator y_iter = Y.begin();
    if (&result == &X)
    {
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            *x_iter = *x_iter * *y_iter + Z;
            x_iter++;
            y_iter++;
        }
    }
    else
    {
        result.clear();
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            result.push_back(*x_iter * *y_iter + Z);
            x_iter++;
            y_iter++;
        }
    }
}

template <typename T>
void np::div(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z)
{
    assert(X.size() == Y.size());
    typename std::vector<T>::iterator x_iter = X.begin();
    typename std::vector<T>::iterator y_iter = Y.begin();
    if (&result == &X)
    {
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            assert(*y_iter != 0);
            *x_iter = *x_iter / *y_iter + Z;
            x_iter++;
            y_iter++;
        }
    }
    else
    {
        result.clear();
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            assert(*y_iter != 0);
            result.push_back(*x_iter / *y_iter + Z);
            x_iter++;
            y_iter++;
        }
    }
}

template <typename T>
void np::add(std::vector<T> &result, std::vector<T> &X, std::vector<T> &Y, T Z)
{
    assert(X.size() == Y.size());
    typename std::vector<T>::iterator x_iter = X.begin();
    typename std::vector<T>::iterator y_iter = Y.begin();
    if (&result == &X)
    {
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            *x_iter = *x_iter + *y_iter + Z;
            x_iter++;
            y_iter++;
        }
    }
    else
    {
        result.clear();

        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            result.push_back(*x_iter + *y_iter + Z);
            x_iter++;
            y_iter++;
        }
    }
}

template <typename T>
void np::add(std::vector<T> &result, std::vector<T> &X, T Y, T Z)
{
    typename std::vector<T>::iterator x_iter = X.begin();
    if (&result == &X)
    {
        while (x_iter != X.end())
        {
            *x_iter = *x_iter + Y + Z;
            x_iter++;
        }
    }
    else
    {
        result.clear();
        while (x_iter != X.end())
        {
            result.push_back(*x_iter + Y + Z);
            x_iter++;
        }
    }
}

template <typename T1, typename T2, typename T3>
void np::sub(std::vector<T1> &result, std::vector<T2> &X, std::vector<T3> &Y)
{
    assert(X.size() == Y.size());
    typename std::vector<T2>::iterator x_iter = X.begin();
    typename std::vector<T3>::iterator y_iter = Y.begin();
    result.clear();
    while ((x_iter != X.end()) && (y_iter != Y.end()))
    {
        // std::cout << "x_iter\t" << static_cast<T1>(*x_iter) << "\ty_iter\t" << static_cast<T2>(*y_iter) << std::endl;
        result.push_back(static_cast<T1>(*x_iter) - static_cast<T1>(*y_iter));
        x_iter++;
        y_iter++;
    }
}

template <typename T>
void np::pow(std::vector<T> &result, std::vector<T> &X, int exp)
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            *iter = static_cast<T>(std::pow(*iter, exp));
        }
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            result.push_back(static_cast<T>(std::pow(*iter, exp)));
        }
    }
}

template <typename T>
void np::sqrt(std::vector<T> &result, std::vector<T> &X)
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            assert(*iter >= 0);
            *iter = static_cast<T>(std::sqrt(*iter));
        }
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            assert(*iter >= 0);
            result.push_back(static_cast<T>(std::sqrt(*iter)));
        }
    }
}

template <typename T>
void np::op2(std::vector<T> &result, std::vector<T> &X, TYPE_VAR (*func)(TYPE_VAR))
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            assert((*iter <= 1) && (*iter >= -1));
            *iter = static_cast<T>(func(*iter));
        }
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            assert((*iter <= 1) && (*iter >= -1));
            result.push_back(static_cast<T>(func(*iter)));
        }
    }
}

template <typename T>
void np::atan(std::vector<T> &result, std::vector<T> &X)
{
    np::op<TYPE_VAR>(result, X, &std::atan);
}

template <typename T>
void np::atan2(std::vector<T> &result, std::vector<T> &Y, std::vector<T> &X)
{
    assert(X.size() == Y.size());
    typename std::vector<T>::iterator x_iter = X.begin();
    typename std::vector<T>::iterator y_iter = Y.begin();
    if (&result == &Y)
    {
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            *y_iter = static_cast<T>(std::atan2(*y_iter, *x_iter));
            x_iter++;
            y_iter++;
        }
    }
    else
    {
        result.clear();
        while ((x_iter != X.end()) && (y_iter != Y.end()))
        {
            assert(*x_iter != 0);
            result.push_back(static_cast<T>(std::atan2(*y_iter, *x_iter)));
            x_iter++;
            y_iter++;
        }
    }
}

template <typename T>
void np::asin(std::vector<T> &result, std::vector<T> &X)
{
    np::op2<TYPE_VAR>(result, X, &std::asin);
}

template <typename T>
void np::acos(std::vector<T> &result, std::vector<T> &X)
{
    np::op2<TYPE_VAR>(result, X, &std::acos);
}

template <typename T>
void np::op(std::vector<T> &result, std::vector<T> &X, TYPE_VAR (*func)(TYPE_VAR))
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            *iter = static_cast<T>(func(*iter));
        }
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            result.push_back(static_cast<T>(func(*iter)));
        }
    }
}

template <typename T>
void np::sin(std::vector<T> &result, std::vector<T> &X)
{
    np::op<TYPE_VAR>(result, X, &std::sin);
}

template <typename T>
void np::cos(std::vector<T> &result, std::vector<T> &X)
{
    np::op<TYPE_VAR>(result, X, &std::cos);
}

template <typename T>
void np::tan(std::vector<T> &result, std::vector<T> &X)
{
    np::op<TYPE_VAR>(result, X, &std::tan);
}

template <typename T, typename T2>
void np::cast(std::vector<T2> &result, std::vector<T> &X)
{
    result.clear();
    for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
    {
        result.push_back(static_cast<T2>(*iter));
    }
}

template <typename T>
void np::mod_1(std::vector<T> &result, std::vector<T> &X)
{
    if (&result == &X)
    {
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            *iter = static_cast<T>(*iter < 0 ? 1 + *iter : *iter);
        }
    }
    else
    {
        result.clear();
        for (typename std::vector<T>::iterator iter = X.begin(); iter != X.end(); iter++)
        {
            result.push_back(static_cast<T>(*iter < 0 ? 1 + *iter : *iter));
        }
    }
}

template <typename T>
void np::take(std::vector<cv::Vec3f> &result, cv::Mat &img, std::vector<T> &Idx)
{
    result.clear();
    cv::Mat temp;
    img.convertTo(temp, CV_32FC3);
    for (typename std::vector<T>::iterator iter = Idx.begin(); iter != Idx.end(); iter++)
    {
        result.push_back(temp.at<cv::Vec3f>(0, *iter));
    }
}

template <typename T>
void np::floor(std::vector<T> &result, std::vector<T> &X)
{
    np::op(result, X, &std::floor);
}

template <typename T>
void np::round(std::vector<T> &result, std::vector<T> &X)
{
    np::op(result, X, &std::round);
}