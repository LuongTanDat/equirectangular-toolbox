#include "vec-360d.h"

template <typename T>
void print(std::vector<T> v, std::string s = "");

template <typename T>
void print(std::vector<T> v, std::string s)
{
    cv::Mat_<T> V = cv::Mat_<T>(v).reshape(1, 1);
    std::cout << s << "\t" << V << std::endl;
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
        // std::cout << "H : " << _shape->H << "  "
        //           << "W : " << _shape->W << "  "
        //           << "C : " << _shape->C << "  "
        //           << cv::Point_<int>(x, y) << std::endl;
        if (pressed)
        {
            // shape *_shape = (shape *)param;
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
    cv::Mat img = cv::imread(std::string("360.jpg"));
    img.copyTo(img_display);
    cv::Mat temp;
    img.convertTo(temp, CV_32FC3);
    shape *_shape = new shape();
    _shape->H = img.rows;
    _shape->W = img.cols;
    _shape->C = img.channels();

    cv::namedWindow("A");
    cv::setMouseCallback("A", callback, (void *)_shape);
    // NFOV nfov = NFOV(3, 4);
    NFOV nfov = NFOV(400, 800);
    while (true)
    {
        // nfov.calcSphericaltoGnomonic(cv::Point2f(0.2, 0.9));
        std::cout << "cp  " << cp << std::endl;
        nfov.calcSphericaltoGnomonic(cp);
        nfov.bilinear_interpolation(img, img_display);

        cv::imshow("A", img_display);

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

    np::mul<TYPE_VAR>(xx_lin, xx_lin, 2, -1);
    np::mul<TYPE_VAR>(yy_lin, yy_lin, 2, -1);

    np::mul<TYPE_VAR>(xx_lin, xx_lin, M_PI * this->FOV.x);
    np::mul<TYPE_VAR>(yy_lin, yy_lin, M_PI_2 * this->FOV.y);

    np::meshgrid<TYPE_VAR>(xx_lin, yy_lin, this->xx, this->yy);
    std::vector<TYPE_VAR> xx_square_2, yy_square_2;
    np::pow<TYPE_VAR>(xx_square_2, this->xx, 2);
    np::pow<TYPE_VAR>(yy_square_2, this->yy, 2);
    np::add<TYPE_VAR>(this->rou, xx_square_2, yy_square_2);
    np::sqrt<TYPE_VAR>(this->rou, this->rou);

    np::atan<TYPE_VAR>(this->c, this->rou);
    np::sin<TYPE_VAR>(this->sin_c, this->c);
    np::cos<TYPE_VAR>(this->cos_c, this->c);
}

void NFOV::calcSphericaltoGnomonic(cv::Point_<TYPE_VAR> center_point)
{
    this->cp = cv::Point_<TYPE_VAR>((center_point.x * 2 - 1) * M_PI, (center_point.y * 2 - 1) * M_PI_2);
    // cout << this->cp << endl;

    this->sin_cp_y = sin(this->cp.y);
    this->cos_cp_y = cos(this->cp.y);

    std::vector<TYPE_VAR> lat, lon;

    std::vector<TYPE_VAR> sin_lat, cos_c_x_cp_y, yy_x_sin_c_rou;
    np::mul<TYPE_VAR>(yy_x_sin_c_rou, this->yy, this->sin_c);
    np::mul<TYPE_VAR>(yy_x_sin_c_rou, yy_x_sin_c_rou, this->cos_cp_y);
    np::div<TYPE_VAR>(yy_x_sin_c_rou, yy_x_sin_c_rou, this->rou);
    np::mul<TYPE_VAR>(cos_c_x_cp_y, this->cos_c, this->sin_cp_y);
    np::add<TYPE_VAR>(sin_lat, yy_x_sin_c_rou, cos_c_x_cp_y);
    np::asin<TYPE_VAR>(this->Lat, sin_lat);

    std::vector<TYPE_VAR> tan_lon, _Y, rou_x_cos_c_x_cos_cp_y, yy_x_sin_cp_y_x_sin_c, _X;
    np::mul<TYPE_VAR>(_Y, this->xx, this->sin_c);
    np::mul<TYPE_VAR>(rou_x_cos_c_x_cos_cp_y, this->rou, this->cos_cp_y);
    np::mul<TYPE_VAR>(rou_x_cos_c_x_cos_cp_y, rou_x_cos_c_x_cos_cp_y, this->cos_c);
    np::mul<TYPE_VAR>(yy_x_sin_cp_y_x_sin_c, this->yy, this->sin_cp_y);
    np::mul<TYPE_VAR>(yy_x_sin_cp_y_x_sin_c, yy_x_sin_cp_y_x_sin_c, this->sin_c);
    np::sub<TYPE_VAR, TYPE_VAR, TYPE_VAR>(_X, rou_x_cos_c_x_cos_cp_y, yy_x_sin_cp_y_x_sin_c);
    np::atan2<TYPE_VAR>(this->Lon, _Y, _X);

    np::mul<TYPE_VAR>(this->Lat, this->Lat, M_1_PI, 0.5);
    np::mul<TYPE_VAR>(this->Lon, this->Lon, M_1_PI / 2, 0.5 + this->cp.x * 0.5 * M_1_PI);
}

void NFOV::bilinear_interpolation(cv::Mat &frame, cv::Mat &result)
{
    if (&frame != &result)
        frame.copyTo(result);
    int frame_height = frame.rows;
    int frame_width = frame.cols;
    int frame_channel = frame.channels();

    std::vector<TYPE_VAR> uf, vf;

    np::mod_1<TYPE_VAR>(uf, this->Lon);
    np::mod_1<TYPE_VAR>(vf, this->Lat);
    np::mul<TYPE_VAR>(uf, uf, frame_width);
    np::mul<TYPE_VAR>(vf, vf, frame_height);

    std::vector<TYPE_VAR> _x0, _y0;
    std::vector<int> x0, y0, x2, y2, base_y0, base_y2, A_idx, B_idx, C_idx, D_idx;

    np::floor<TYPE_VAR>(_x0, uf);
    np::floor<TYPE_VAR>(_y0, vf);
    np::cast<TYPE_VAR, int>(x0, _x0);
    np::cast<TYPE_VAR, int>(y0, _y0);
    np::add<int>(x2, x0, 1);
    np::add<int>(y2, y0, 1);
    np::mul<int>(base_y0, y0, frame_width);
    np::mul<int>(base_y2, y2, frame_width);
    np::add<int>(A_idx, base_y0, x0);
    np::add<int>(B_idx, base_y2, x0);
    np::add<int>(C_idx, base_y0, x2);
    np::add<int>(D_idx, base_y2, x2);

    result = result.reshape(3, 1);
    std::vector<TYPE_VAR> wa, wa_1, wa_2, wb, wb_1, wb_2, wc, wc_1, wc_2, wd, wd_1, wd_2;
    np::sub<TYPE_VAR, int, TYPE_VAR>(wa_1, x2, uf);
    np::sub<TYPE_VAR, int, TYPE_VAR>(wa_2, y2, vf);
    np::mul<TYPE_VAR>(wa, wa_1, wa_2);
    np::sub<TYPE_VAR, int, TYPE_VAR>(wb_1, x2, uf);
    np::sub<TYPE_VAR, TYPE_VAR, int>(wb_2, vf, y0);
    np::mul<TYPE_VAR>(wb, wb_1, wb_2);
    np::sub<TYPE_VAR, TYPE_VAR, int>(wc_1, uf, x0);
    np::sub<TYPE_VAR, int, TYPE_VAR>(wc_2, y2, vf);
    np::mul<TYPE_VAR>(wc, wc_1, wc_2);
    np::sub<TYPE_VAR, TYPE_VAR, int>(wd_1, uf, x0);
    np::sub<TYPE_VAR, TYPE_VAR, int>(wd_2, vf, y0);
    np::mul<TYPE_VAR>(wd, wd_1, wd_2);

    std::vector<cv::Vec3f> A, B, C, D;
    np::take<int>(A, frame, A_idx);
    np::take<int>(B, frame, B_idx);
    np::take<int>(C, frame, C_idx);
    np::take<int>(D, frame, D_idx);
    std::vector<cv::Vec3f>::iterator A_iter = A.begin();
    std::vector<cv::Vec3f>::iterator B_iter = B.begin();
    std::vector<cv::Vec3f>::iterator C_iter = C.begin();
    std::vector<cv::Vec3f>::iterator D_iter = D.begin();
    std::vector<TYPE_VAR>::iterator wa_iter = wa.begin();
    std::vector<TYPE_VAR>::iterator wb_iter = wb.begin();
    std::vector<TYPE_VAR>::iterator wc_iter = wc.begin();
    std::vector<TYPE_VAR>::iterator wd_iter = wd.begin();
    cv::Mat temp = cv::Mat(1, this->height * this->width, CV_32FC3, 0.0f);

    int cnt = 0;
    while (A_iter != A.end())
    {
        cv::Vec3f AA = *A_iter * *wa_iter;
        cv::Vec3f BB = *B_iter * *wb_iter;
        cv::Vec3f CC = *C_iter * *wc_iter;
        cv::Vec3f DD = *D_iter * *wd_iter;
        temp.at<cv::Vec3f>(0, cnt++) = (AA + BB + CC + DD);
        A_iter++;
        B_iter++;
        C_iter++;
        D_iter++;
        wa_iter++;
        wb_iter++;
        wc_iter++;
        wd_iter++;
    }

    temp = temp.reshape(3, this->height);
    temp.convertTo(result, CV_8UC3);
}