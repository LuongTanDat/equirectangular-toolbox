#include "cv-360d.h"

template <typename T>
void print(std::vector<T> v, std::string s = "");

template <typename T>
void print(T v, std::string s = "");

template <typename T>
void print(std::vector<T> v, std::string s)
{
    cv::Mat_<T> V = cv::Mat_<T>(v).reshape(1, 1);
    std::cout << s << "\t" << V << std::endl;
}

template <typename T>
void print(T v, std::string s)
{
    std::cout << s << "\t" << v << std::endl;
}

cv::Point_<TYPE_VAR> cp = cv::Point_<TYPE_VAR>(0., 0.);
cv::Point_<int> point = cv::Point_<int>(0, 0);
cv::Point_<TYPE_VAR> delta_position = cv::Point_<TYPE_VAR>(0., 0.);
bool pressed = false;

using namespace std;

struct shape
{
    int H, W, C;
};

void callback(int event, int x, int y, int, void *param)
{
    if (event == cv::EVENT_MOUSEMOVE) //finding first corner
    {
        shape *_shape = (shape *)param;
        if (pressed)
        {
            cp = cv::Point_<TYPE_VAR>(cp.x + static_cast<TYPE_VAR>(point.x - x) / static_cast<TYPE_VAR>(_shape->W), cp.y + static_cast<TYPE_VAR>(point.y - y) / static_cast<TYPE_VAR>(_shape->H));
            point = cv::Point_<int>(x, y);
        }
    }
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        pressed = true;
        point = cv::Point_<int>(x, y);
    }

    else if (event == cv::EVENT_LBUTTONUP) //tracking mouse movement
    {
        pressed = false;
    }
}

int main()
{
    cv::Mat img_display;
    // cv::Mat img = cv::imread(std::string("360.jpg"));
    cv::Mat img = cv::imread(std::string("FAVSA.png"));
    img.copyTo(img_display);
    shape *_shape = new shape();
    _shape->H = img.rows;
    _shape->W = img.cols;
    _shape->C = img.channels();

    cv::namedWindow("C++");
    cv::setMouseCallback("C++", callback, (void *)_shape);
    // NFOV nfov = NFOV(3, 4);
    NFOV nfov = NFOV(400, 800);
    clock_t tStart;
    while (true)
    {
        // nfov.calcSphericaltoGnomonic(cv::Point2f(0.2, 0.9));
        nfov.calcSphericaltoGnomonic(cp);
        nfov.bilinear_interpolation(img, img_display);

        cv::imshow("C++", img_display);
        unsigned char k = uchar(cv::waitKey(1));
        if (k == 27)
            break;
    }
    return 0;
}

NFOV::NFOV(int height, int width)
{
    this->set_screen_points(height, width);
}

void NFOV::set_screen_points(int height, int width)
{
    this->height = height;
    this->width = width;

    std::vector<TYPE_VAR> xx_lin, yy_lin;
    np::linspace<TYPE_VAR, TYPE_VAR>(0, 1, this->width, xx_lin);
    np::linspace<TYPE_VAR, TYPE_VAR>(0, 1, this->height, yy_lin);

    cv::Mat_<TYPE_VAR> xx_mat, yy_mat;
    xx_mat = cv::Mat_<TYPE_VAR>(xx_lin).reshape(1, 1);
    yy_mat = cv::Mat_<TYPE_VAR>(yy_lin).reshape(1, 1);
    cv::multiply(xx_mat, 2, xx_mat);
    cv::multiply(yy_mat, 2, yy_mat);
    cv::subtract(xx_mat, 1, xx_mat);
    cv::subtract(yy_mat, 1, yy_mat);
    cv::multiply(xx_mat, M_PI * this->FOV.x, xx_mat);
    cv::multiply(yy_mat, M_PI_2 * this->FOV.y, yy_mat);
    np::meshgrid<TYPE_VAR>(xx_mat, yy_mat, this->xx, this->yy);

    cv::Mat_<TYPE_VAR> xx_square_2, yy_square_2;
    cv::multiply(this->xx, this->xx, xx_square_2);
    cv::multiply(this->yy, this->yy, yy_square_2);
    cv::add(xx_square_2, yy_square_2, this->rou);
    cv::sqrt(this->rou, this->rou);

    np::atan<TYPE_VAR>(this->c, this->rou);
    np::sin<TYPE_VAR>(this->sin_c, this->c);
    np::cos<TYPE_VAR>(this->cos_c, this->c);
}

void NFOV::calcSphericaltoGnomonic(cv::Point_<TYPE_VAR> center_point)
{
    this->cp = cv::Point_<TYPE_VAR>((center_point.x * 2 - 1) * M_PI, (center_point.y * 2 - 1) * M_PI_2);

    this->sin_cp_y = std::sin(this->cp.y);
    this->cos_cp_y = std::cos(this->cp.y);

    cv::Mat_<TYPE_VAR> lat, lon, sin_lat, cos_c_x_cp_y, yy_x_sin_c_rou;

    cv::multiply(this->yy, this->sin_c, yy_x_sin_c_rou);
    cv::multiply(yy_x_sin_c_rou, this->cos_cp_y, yy_x_sin_c_rou);
    cv::divide(yy_x_sin_c_rou, this->rou, yy_x_sin_c_rou);
    cv::multiply(this->cos_c, this->sin_cp_y, cos_c_x_cp_y);
    cv::add(yy_x_sin_c_rou, cos_c_x_cp_y, sin_lat);
    np::asin<TYPE_VAR>(this->Lat, sin_lat);

    cv::Mat_<TYPE_VAR> tan_lon, _Y, rou_x_cos_c_x_cos_cp_y, yy_x_sin_cp_y_x_sin_c, _X;
    cv::multiply(this->xx, this->sin_c, _Y);
    cv::multiply(this->rou, this->cos_cp_y, rou_x_cos_c_x_cos_cp_y);
    cv::multiply(rou_x_cos_c_x_cos_cp_y, this->cos_c, rou_x_cos_c_x_cos_cp_y);
    cv::multiply(this->yy, this->sin_cp_y, yy_x_sin_cp_y_x_sin_c);
    cv::multiply(yy_x_sin_cp_y_x_sin_c, this->sin_c, yy_x_sin_cp_y_x_sin_c);
    cv::subtract(rou_x_cos_c_x_cos_cp_y, yy_x_sin_cp_y_x_sin_c, _X);

    np::atan2<TYPE_VAR>(this->Lon, _Y, _X);

    cv::multiply(this->Lat, M_1_PI, this->Lat);
    cv::add(this->Lat, 0.5, this->Lat);
    cv::multiply(this->Lon, M_1_PI * 0.5, this->Lon);
    cv::add(this->Lon, 0.5 + this->cp.x * 0.5 * M_1_PI, this->Lon);
}

void NFOV::bilinear_interpolation(cv::Mat &frame, cv::Mat &result)
{
    if (&frame != &result)
        frame.copyTo(result);
    int frame_height = frame.rows;
    int frame_width = frame.cols;
    int frame_channel = frame.channels();

    cv::Mat_<TYPE_VAR> uf, vf;
    np::mod_1<TYPE_VAR>(uf, this->Lon);
    np::mod_1<TYPE_VAR>(vf, this->Lat);
    cv::multiply(uf, frame_width, uf);
    cv::multiply(vf, frame_height, vf);
    cv::Mat_<TYPE_VAR> _x0, _y0;
    cv::Mat_<TYPE_VAR> x0, y0, x2, y2, base_y0, base_y2, A_idx, B_idx, C_idx, D_idx;
    np::floor<TYPE_VAR>(x0, uf);
    np::floor<TYPE_VAR>(y0, vf);
    // _x0.convertTo(x0, CV_32S);
    // _y0.convertTo(y0, CV_32S);
    cv::add(x0, 1, x2);
    cv::add(y0, 1, y2);
    cv::multiply(y0, frame_width, base_y0);
    cv::multiply(y2, frame_width, base_y2);
    cv::add(base_y0, x0, A_idx);
    cv::add(base_y2, x0, B_idx);
    cv::add(base_y0, x2, C_idx);
    cv::add(base_y2, x2, D_idx);

    result = result.reshape(3, 1);
    cv::Mat_<TYPE_VAR> wa, wa_1, wa_2, wb, wb_1, wb_2, wc, wc_1, wc_2, wd, wd_1, wd_2;
    // cv::multiply(x2 - uf, y2 - vf, wa);
    cv::subtract(x2, uf, wa_1);
    cv::subtract(y2, vf, wa_2);
    cv::multiply(wa_1, wa_2, wa);
    cv::subtract(x2, uf, wb_1);
    cv::subtract(vf, y0, wb_2);
    cv::multiply(wb_1, wb_2, wb);
    cv::subtract(uf, x0, wc_1);
    cv::subtract(y2, vf, wc_2);
    cv::multiply(wc_1, wc_2, wc);
    cv::subtract(uf, x0, wd_1);
    cv::subtract(vf, y0, wd_2);
    cv::multiply(wd_1, wd_2, wd);
    // print(wa, "wa");
    // print(wb, "wb");
    // print(wc, "wc");
    // print(wd, "wd");

    cv::Mat_<cv::Vec<TYPE_VAR, 3>> AA = cv::Mat_<cv::Vec<TYPE_VAR, 3>>(this->width, this->height).reshape(1, 1);
    cv::Mat_<cv::Vec<TYPE_VAR, 3>> BB = cv::Mat_<cv::Vec<TYPE_VAR, 3>>(this->width, this->height).reshape(1, 1);
    cv::Mat_<cv::Vec<TYPE_VAR, 3>> CC = cv::Mat_<cv::Vec<TYPE_VAR, 3>>(this->width, this->height).reshape(1, 1);
    cv::Mat_<cv::Vec<TYPE_VAR, 3>> DD = cv::Mat_<cv::Vec<TYPE_VAR, 3>>(this->width, this->height).reshape(1, 1);
    np::take<int>(AA, frame, A_idx, wa);
    np::take<int>(BB, frame, B_idx, wb);
    np::take<int>(CC, frame, C_idx, wc);
    np::take<int>(DD, frame, D_idx, wd);

    cv::Mat_<cv::Vec<TYPE_VAR, 3>> tmp = AA + BB + CC + DD;
    tmp = tmp.reshape(3, this->height);
    tmp.convertTo(result, CV_8UC3);
}
