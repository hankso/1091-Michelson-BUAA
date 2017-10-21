#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:45:43 2017

@author: hank

for 1091 Michelson Experiment, BUAA
"""

import cv2, time

class Physics_1091(object):
    def __init__(self,
                 fps = 25,
                 winname = '1091 Michelson -- auto reading cycles'):

        self._opencam()

        self.winname = winname
        self.counter = 0
        self.text = ''
        self.fps = fps
        self._aoi = None
        self._rsw = False

        self.c_green = (10, 255, 10)
        self.c_red = (10, 10, 100)
        self.c_white = (255, 255, 255)

        #          LP1,LP2,RP1,RP2  LD, RD      text_org_point
        self._info = [(0, 0)] * 4 + [False] * 2 + [(10, 30)]


    def _opencam(self):
        self.cam = cv2.VideoCapture(0)
        for _ in xrange(3):
            if self.cam.isOpened():
                print('camera is opened')
                for _ in xrange(50):
                    self.cam.read()
                self.frame = self.cam.read()[1]
                return True
            try:
                self.cam.open()
            except:
                print('camera is wrong')
                time.sleep(2)
        print('terminated.')
        quit()

    def _mouse(self, event, x, y, flags, param):
# =============================================================================
#         x, y = int(x*self._r), int(y*self._r)
# =============================================================================
        if event == cv2.EVENT_LBUTTONDOWN:
            self._info[4] = True
            self._info[0] = (x, y)
            if flags == cv2.EVENT_FLAG_CTRLKEY:
                self._aoi = None
        elif event == cv2.EVENT_LBUTTONUP:
            self._info[4] = False
            self._info[1] = (x, y)
            if self._info[0] != self._info[1]:
                x1, y1 = self._info[0]
                t = (min(y1, y),
                     max(y1, y),
                     min(x1, x),
                     max(x1, x))
                if flags == cv2.EVENT_FLAG_CTRLKEY + 1:
                    self._aoi = t
                else:
                    self._rsw = t
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._info[5] = True
            self._info[2] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._info[5] = False
            self._info[3] = (x, y)
            if (x, y) <= self._info[2]:
                self._aoi = None
                self._rsw = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._info[4]:
                self._info[1] = (x, y)
                cv2.rectangle(self.frame,
                              self._info[0],
                              self._info[1],
                              self.c_red,
                              1, cv2.LINE_AA)
            if self._info[5]: self._info[3] = (x, y)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._aoi = None
            self._rsw = False

    def process_img(self, img = None, rsw = None, aoi = None):
        if not img: img = self.frame
        if not rsw: rsw = self._rsw
        if not aoi: aoi = self._aoi

# =============================================================================
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# =============================================================================

        if rsw:
            cv2.imshow('src', cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img))))
            # rsw: ymin, ymax, xmin, xmax
            H, W = img.shape[:2]
            h, w = rsw[1]-rsw[0], rsw[3]-rsw[2]
            ratio = min(W/w, H/h)
            img = img[rsw[0]:rsw[1], rsw[2]:rsw[3]]
            img = cv2.resize(img, None, fx = ratio, fy = ratio)
        else: cv2.destroyWindow('src')

        if aoi:
            # aoi: ymin, ymax, xmin, xmax
            cv2.rectangle(img,
                          (aoi[2], aoi[0]),
                          (aoi[3], aoi[1]),
                          self.c_green,
                          1, cv2.LINE_AA)

        return img

    def run(self):
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self._mouse)

        while self.cam.isOpened():

            t = time.time()


            cv2.imshow(self.winname, self.frame)
            k = cv2.waitKey(int(1000/self.fps))
            self.frame = self.cam.read()[1]
            self.frame = self.process_img()
            cv2.putText(self.frame, self.text, self._info[6],
                        cv2.FONT_HERSHEY_DUPLEX, 0.6,
                        self.c_white, 1, cv2.LINE_AA)
            if k in [113, 27]:
                cv2.destroyAllWindows()
                self.cam.release()
            elif k == ord('t'):
                self._info[6] = self._info[0]

            # t: 0.1s (for example)
            t = time.time() - t
            self.text = '{} -fps: {} -cycles: {}'.format(self.winname, int(1/t), self.counter)


if __name__ == '__main__':
    p = Physics_1091()
    p.run()
