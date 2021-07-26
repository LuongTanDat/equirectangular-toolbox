// #include "cv-360d.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#define TEST_EXECUTION_TIME false
// #define DEBUG1

namespace py = boost::python;
namespace np = boost::python::numpy;

typedef float TYPE_VAR;

cv::Point_<double> cp = cv::Point_<double>(0., 0.);
cv::Point_<int> point = cv::Point_<int>(0, 0);
cv::Point_<TYPE_VAR> delta_position = cv::Point_<TYPE_VAR>(0., 0.);
bool pressed = false;

void Init()
{
    // Set your python location.
    // wchar_t str[] = L"/home/vietanhdev/miniconda3/envs/example_env";
    // Py_SetPythonHome(str);

    setenv("PYTHONPATH", ".", 1);

    Py_Initialize();
    np::initialize();
}

// Function to convert from cv::Mat to numpy array
np::ndarray ConvertMatToNDArray(cv::Mat &mat)
{
    py::tuple shape = py::make_tuple(mat.rows, mat.cols, mat.channels());
    py::tuple stride = py::make_tuple(mat.channels() * mat.cols * sizeof(uchar), mat.channels() * sizeof(uchar), sizeof(uchar));
    np::dtype dt = np::dtype::get_builtin<uchar>();
    np::ndarray ndImg = np::from_data(mat.data, dt, shape, stride, py::object());
    return ndImg;
}

np::ndarray ConvertPoint2dToNDArray(cv::Point_<double> &point)
{
    py::tuple shape = py::make_tuple(2);
    py::tuple stride = py::make_tuple(sizeof(double));
    np::dtype dt = np::dtype::get_builtin<double>();
    double point2d[2] = {point.x, point.y};
    np::ndarray pyPoint = np::from_data((void *)point2d, dt, shape, stride, py::object());
    return pyPoint;
}

// Function to convert from numpy array to cv::Mat
cv::Mat ConvertNDArrayToMat(const np::ndarray &ndarr)
{
    int length = ndarr.get_nd();                  // get_nd() returns num of dimensions. this is used as
                                                  // a length, but we don't need to use in this case.
                                                  // because we know that image has 3 dimensions.
    const Py_intptr_t *shape = ndarr.get_shape(); // get_shape() returns Py_intptr_t* which we can get
                                                  // the size of n-th dimension of the ndarray.
    char *dtype_str = py::extract<char *>(py::str(ndarr.get_dtype()));

    // Variables for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = length == 3 ? shape[2] : 1;
    int depth;

    // Find corresponding datatype in C++
    if (!strcmp(dtype_str, "uint8"))
    {
        depth = CV_8U;
    }
    else if (!strcmp(dtype_str, "int8"))
    {
        depth = CV_8S;
    }
    else if (!strcmp(dtype_str, "uint16"))
    {
        depth = CV_16U;
    }
    else if (!strcmp(dtype_str, "int16"))
    {
        depth = CV_16S;
    }
    else if (!strcmp(dtype_str, "int32"))
    {
        depth = CV_32S;
    }
    else if (!strcmp(dtype_str, "float32"))
    {
        depth = CV_32F;
    }
    else if (!strcmp(dtype_str, "float64"))
    {
        depth = CV_64F;
    }
    else
    {
        std::cout << "Wrong dtype error" << std::endl;
        return cv::Mat();
    }

    int type = CV_MAKETYPE(depth, channel); // Create specific datatype using channel information
    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);
    return mat;
}

struct shape
{
    shape(int H, int W, int C) : H(H), W(W), C(C)
    {
    }

    int H, W, C;
};

void callback(int event, int x, int y, int, void *param)
{
    if (event == cv::EVENT_MOUSEMOVE) //finding first corner
    {
        shape *_shape = (shape *)param;
        if (pressed)
        {
#ifdef DEBUG1
            std::cout << "LINE 124\t" << cp.x << "\t" << cp.y << std::endl;
#endif // DEBUG1
            cp = cv::Point_<TYPE_VAR>(cp.x + static_cast<TYPE_VAR>(point.x - x) / static_cast<TYPE_VAR>(_shape->W), cp.y + static_cast<TYPE_VAR>(point.y - y) / static_cast<TYPE_VAR>(_shape->H));
#ifdef DEBUG1
            std::cout << "LINE 126\t" << cp.x << "\t" << cp.y << std::endl;
#endif // DEBUG1
            point = cv::Point_<int>(x, y);
        }
    }
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        pressed = true;
        // std::cout << "pressed = true;" << std::endl;
        point = cv::Point_<int>(x, y);
    }

    else if (event == cv::EVENT_LBUTTONUP) //tracking mouse movement
    {
        pressed = false;
        // std::cout << "pressed = false;" << std::endl;
    }
}

int main()
{
    setlocale(LC_ALL, "");
    try
    {
        // Initialize boost python and numpy
        Init();
        // Import module
        py::object main_module = py::import("__main__");
        // Load the dictionary for the namespace
        py::object mn = main_module.attr("__dict__");
        // Import the module into the namespace
        py::exec("import dewraping", mn);

        // Create the locally-held object
        py::object NFOV = py::eval("dewraping.NFOV()", mn);
        py::object nfovInit = NFOV.attr("Init");
        py::object toNFOV = NFOV.attr("toNFOV");

        cv::Mat img_display;
        cv::Mat img = cv::imread(std::string("360.jpg"));
        // cv::Mat img = cv::imread(std::string("FAVSA.png"));
        // img.copyTo(img_display);
        shape _shape = shape(img.rows, img.cols, img.channels());
        cv::namedWindow("C++");
        cv::setMouseCallback("C++", callback, (void *)&_shape);

        py::extract<np::ndarray>(nfovInit(600, 1200));

        while (1)
        {
#ifdef DEBUG1
            std::cout << "" << cp.x << "\t" << cp.y << std::endl;
#endif // DEBUG1
            np::ndarray nd_img = ConvertMatToNDArray(img);
#ifdef DEBUG1
            std::cout << py::extract<char const *>(py::str(py_point)) << std::endl;
#endif // DEBUG1
            np::ndarray output_img = py::extract<np::ndarray>(toNFOV(nd_img, cp.x, cp.y));
            // std::cout << py::extract<char const *>(py::str(output_img)) << std::endl;
            cv::Mat img_display = ConvertNDArrayToMat(output_img);
            cv::imshow("C++", img_display);
            unsigned char k = uchar(cv::waitKey(1));
            if (k == 27)
                break;
        }
    }
    catch (py::error_already_set &)
    {
        PyErr_Print();
    }

    return 0;
}
