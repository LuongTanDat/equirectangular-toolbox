# Copyright 2017 Nitish Mutha (nitishmutha.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from math import pi
import numpy as np
import cv2

cp = [0., 0.]
point = [0, 0]
pressed = False
delta_position = [0., 0.]


class NFOV():
    def __init__(self):
        pass

    def Init(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self._set_screen_points(height=height, width=width)

    def _set_screen_points(self, height, width):
        self.height = height
        self.width = width

        self._screen_points = self._get_screen_img()
        self._sp = (self._screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * \
            (np.ones(self._screen_points.shape) * self.FOV)
        self.x = self.sp.T[0]
        self.y = self.sp.T[1]
        self.rou = np.sqrt(self.x ** 2 + self.y ** 2)
        self.c = np.arctan(self.rou)
        self.sin_c = np.sin(self.c)
        self.cos_c = np.cos(self.c)

    @property
    def sp(self):
        return self._sp

    def _get_coord_rad(self, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2])

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width),
                             np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self):
        # latitude
        lat = np.arcsin(
            self.cos_c * np.sin(self.cp[1]) + (self.y * self.sin_c * np.cos(self.cp[1])) / self.rou)
        # longtitude
        lon = self.cp[0] + np.arctan2(self.x * self.sin_c,
                                      self.rou * np.cos(self.cp[1]) * self.cos_c -
                                      self.y * np.sin(self.cp[1]) * self.sin_c)
        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5
        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0], 1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1], 1) * self.frame_height  # lat - height
        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        # coords of pixel to top right
        x2 = np.add(x0, np.ones(uf.shape).astype(int))
        y2 = np.add(y0, np.ones(vf.shape).astype(int))
        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)
        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)
        flat_img = np.reshape(self.frame, [-1, self.frame_channel])
        A = np.take(flat_img, A_idx, axis=0, mode="clip")
        B = np.take(flat_img, B_idx, axis=0, mode="clip")
        C = np.take(flat_img, C_idx, axis=0, mode="clip")
        D = np.take(flat_img, D_idx, axis=0, mode="clip")
        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)
        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(
            np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        return nfov

    def toNFOV(self, frame, xc, yc):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = self._get_coord_rad(center_point=np.array((xc, yc)))
        spericalCoord = self._calcSphericaltoGnomonic()
        return self._bilinear_interpolation(spericalCoord)


def callback(event, x, y, flags, param):
    global cp, point, pressed, delta_position
    if event == cv2.EVENT_MOUSEMOVE:
        if pressed:
            cp = [cp[0] + (point[0] - x) / param[1],
                  cp[1] + (point[1] - y) / param[0]]
            point = [x, y]
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        point = [x, y]
    if event == cv2.EVENT_LBUTTONUP:
        pressed = False


# test the class
if __name__ == '__main__':
    img = cv2.imread("360.jpg")
    # img = cv2.imread("FAVSA.png")
    cv2.namedWindow("PYTHON", cv2.WINDOW_NORMAL)
    nfov = NFOV()
    nfov.Init(height=400, width=800)
    cv2.setMouseCallback("PYTHON", callback, param=img.shape)
    while True:
        # camera center point (valid range [0,1])
        fov_img = nfov.toNFOV(img, cp[0], cp[1])
        cv2.imshow("PYTHON", fov_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
