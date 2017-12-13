#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:45:43 2017
@author: hank
for 1091 Michelson Interfere of 'Basic Physical Experiments', BUAA
"""
import cv2, time, numpy as np, sys

class BPE(object):
    def __init__(self,
                 src = None,
                 fps = 25,
                 winname = '1091 Michelson Interfere -- auto-count fringes',
                 ocr = False):

        self.winname = winname
        self.fringe_counter = 0
        self.text = ''
        self._fps = fps
        self._ftd = int(1000./self._fps)

        self._open_cam(src)

        self._ROI = None
        self._RSW = None

        self._mor_e = 0
        self._mor_d = 0
        self._contours_size = 0

        self.c_green = ( 10, 255,  10)
        self.c_red   = (  5,   5, 220)
        self.c_blue  = (130,  15,  10)
        self.c_white = (255, 255, 255)
        self.c_black = (  5,   5,   5)

        self._img_pick = np.repeat([0,0,0],180*60,0).reshape(60,180,3).astype(np.uint8)
        self._img_hsvu = np.repeat([0,0,0],180*60,0).reshape(60,180,3).astype(np.uint8)
        self._img_hsvd = np.repeat([0,0,0],180*60,0).reshape(60,180,3).astype(np.uint8)

        cv2.namedWindow(self.winname, cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(self.winname, self._mouse)
        cv2.imshow(self.winname, np.zeros((20,20),np.uint8))
        cv2.namedWindow('bar')
        cv2.createTrackbar('H d', 'bar', 0, 180, self._on_hd)
        cv2.createTrackbar('S d', 'bar', 40, 255, self._on_sd)
        cv2.createTrackbar('V d', 'bar', 40, 255, self._on_vd)
        cv2.createTrackbar('H u', 'bar', 90, 180, self._on_hu)
        cv2.createTrackbar('S u', 'bar', 150, 255, self._on_su)
        cv2.createTrackbar('V u', 'bar', 200, 255, self._on_vu)
        cv2.createTrackbar('erode', 'bar', 0,  10, self._on_me)
        cv2.createTrackbar('dilate', 'bar', 0,  10, self._on_md)
        cv2.createTrackbar('c_size', 'bar', 0, 9999, self._on_cs)


        self._info = []
        #             LP1,LP2,RP1,RP2   LD,RD,LP,RP
        self._info += [(0, 0)] * 4   +  [False] * 4
        #             text_point        DLU,DRD
        self._info += [(10, 30)]     +  [(0, 0)] * 2
        #             D,FD,EP
        self._info += [False] * 3
        # i.e. info = [ LeftPoint1,      tuple         0
        #               LeftPoint2,      tuple         1
        #               RightPoint1,     tuple         2
        #               RightPoint2,     tuple         3
        #               LeftDown,        bool          4
        #               RightDown,       bool          5
        #               LeftPaint,       bool          6
        #               RightPaint,      bool          7
        #               TextPoint,       tuple         8
        #               DetectLeftUp,    tuple         9
        #               DetectRightDown, tuple         10
        #               Detected,        bool          11
        #               Flag_Detect,     bool          12
        #               EffectPallet,    bool          13


        if ocr:
            import keras
            self.ocr_model = keras.models.load_model('CNN-with-weights-4.9M-20161102.h5')
            self.start_time = time.time()
            self.num_list = []
            self._FLAG_OCR = True
        else: self._FLAG_OCR = False

    def _on_hu(self, x):
        self._img_hsvu[:,:,0] = x
    def _on_su(self, x):
        self._img_hsvu[:,:,1] = x
    def _on_vu(self, x):
        self._img_hsvu[:,:,2] = x
    def _on_hd(self, x):
        self._img_hsvd[:,:,0] = x
    def _on_sd(self, x):
        self._img_hsvd[:,:,1] = x
    def _on_vd(self, x):
        self._img_hsvd[:,:,2] = x
    def _on_me(self, x):
        self._mor_e = x #(2*x+1, 2*x+1)
    def _on_md(self, x):
        self._mor_d = x #(2*x+1, 2*x+1)
    def _on_cs(self, x):
        self._contours_size = x

    def _open_cam(self, src):
        if src is None:
            self.cam = cv2.VideoCapture(src)
            for _ in xrange(3):
                if self.cam.isOpened():
                    print('camera is opened')
                    for _ in xrange(20):
                        self.cam.read()
                    self.frame = self.cam.read()[1]
                    return True
                try:
                    self.cam.open()
                except:
                    print('camera is wrong')
                    time.sleep(2)
            print('terminated.')
            sys.exit()
        elif isinstance(src, str):
            if src[-3:] in ['avi', 'mp4']:
                self.cam = cv2.VideoCapture(src)
                return True
            try:
                self.cam = _virtual_cam_class(src)
                self._fps = 5
                self._ftd = 200
                return True
            except: pass
        print('Source {} is not supported.\nAbort.'.format(src))
        sys.exit()

    def _mouse(self, event, x, y, flags, param):
        # x, y = int(x*self._r), int(y*self._r)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._info[4] = True
            self._info[0] = (x, y)
            if flags == cv2.EVENT_FLAG_CTRLKEY:
                self._img_pick[:,:] = self.frame[y, x]
                b, g, r = self.frame[y, x]
                h, s, v = cv2.cvtColor(self.frame[y, x].reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]
                cv2.putText(self._img_pick, 'H: %3d S: %3d V: %3d'%(h, s, v), (5, 15),
                            cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)
                cv2.putText(self._img_pick, 'R: %3d G: %3d B: %3d'%(r, g, b), (5, 35),
                            cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)
            if flags == cv2.EVENT_FLAG_ALTKEY:
                self._info[9] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._info[4] = False
            self._info[6] = False
            self._info[1] = (x, y)
            if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:
                if self._info[0] != self._info[1]:
                    x1, y1 = self._info[0]
                    self._ROI = (min(y1, y), max(y1, y), min(x1, x), max(x1, x))
            elif flags == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_LBUTTONDOWN:
                if self._info[9] != (x, y):
                    x1, y1 = self._info[9]
                    self._info[9]  = min(x1, x), min(y1, y)
                    self._info[10] = max(x1, x), max(y1, y)
            elif flags == 0:
                if self._info[0] != self._info[1]:
                    x1, y1 = self._info[0]
                    self._RSW = (min(y1, y), max(y1, y), min(x1, x), max(x1, x))
                    self._ROI = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._info[5] = True
            self._info[2] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._info[5] = False
            self._info[7] = False
            self._info[3] = (x, y)
            if self._info[2] != self._info[3]:
                x1, y1 = self._info[2]
                x2, y2 = self._info[3]
                if x1 > x2 and y1 > y2:
                    self._ROI = None
                    self._RSW = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._info[4]:
                self._info[1] = (x, y)
                self._info[6] = True
            if self._info[5]:
                self._info[3] = (x, y)
                self._info[7] = True
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._ROI = None
            self._RSW = None
        else: print(event, flags, x, y, param)

    def _keyboard(self, k):
        # if k != 255: print('key: %d'%k)
        if k in [113, 27]:  # Quit or Esc
            time.sleep(0.5)
            cv2.destroyAllWindows()
            self.cam.release()
        elif k == ord('m'): # Move putText position
            self._info[6] = self._info[0]
        elif k == ord('p'): # Pause
            self._pause_display()
        elif k == ord('e'): # Effect
            self._info[13] = not self._info[13]
        elif k == ord('d'): # Detect
            self._info[12] = not self._info[12]

    def _pause_display(self):
        pos = np.array(self.frame.shape[:2][::-1])/2 + [-100, 0]
        cv2.putText(self.frame, 'Pause', tuple(pos),
                    cv2.FONT_HERSHEY_DUPLEX, 2, self.c_black, 3, cv2.LINE_AA)
        cv2.putText(self.frame, 'Click enter to resume', tuple(pos+[-250, 80]),
                    cv2.FONT_HERSHEY_DUPLEX, 2, self.c_black, 3, cv2.LINE_AA)
        cv2.putText(self.frame, 'Pause', tuple(pos),
                    cv2.FONT_HERSHEY_DUPLEX, 2, self.c_white, 2, cv2.LINE_AA)
        cv2.putText(self.frame, 'Click enter to resume', tuple(pos+[-250, 80]),
                    cv2.FONT_HERSHEY_DUPLEX, 2, self.c_white, 2, cv2.LINE_AA)
        cv2.imshow(self.winname, self.frame)
        while cv2.waitKey() != ord('\n'): pass

    def _update_bar_window(self):
        hsvu_bgr = cv2.cvtColor(self._img_hsvu, cv2.COLOR_HSV2BGR)
        hu, su, vu = self._img_hsvu[0, 0]
        bu, gu, ru = hsvu_bgr[0, 0]
        cv2.putText(hsvu_bgr, 'H: %3d S: %3d V: %3d'%(hu, su, vu), (5, 15),
                    cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)
        cv2.putText(hsvu_bgr, 'R: %3d G: %3d B: %3d'%(ru, gu, bu), (5, 35),
                    cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)

        hsvd_bgr = cv2.cvtColor(self._img_hsvd, cv2.COLOR_HSV2BGR)
        hd, sd, vd = self._img_hsvd[0, 0]
        bd, gd, rd = hsvd_bgr[0, 0]
        cv2.putText(hsvd_bgr, 'H: %3d S: %3d V: %3d'%(hd, sd, vd), (5, 15),
                    cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)
        cv2.putText(hsvd_bgr, 'R: %3d G: %3d B: %3d'%(rd, gd, bd), (5, 35),
                    cv2.FONT_ITALIC, 0.5, self.c_white, 1, cv2.LINE_AA)

        cv2.imshow('bar', np.vstack((hsvd_bgr,
                                     self._img_pick,
                                     hsvu_bgr)))

    def _update_effect_window(self, roi):
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._img_hsvd[0,0], self._img_hsvu[0,0])
        for _ in xrange(self._mor_e):
            mask = cv2.erode(mask, (3, 3))
        for _ in xrange(self._mor_d):
            mask = cv2.dilate(mask, (3, 3))
        rst  = cv2.bitwise_and(self.roi, self.roi, mask = mask)
        contours = np.array(cv2.findContours(mask,
                                             cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)[1])
        cv2.drawContours(roi,
                         contours[np.array([cv2.contourArea(_) for _ in contours]) > self._contours_size],
                         -1, self.c_blue)
        cv2.imshow('effect', np.vstack((roi,
                                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                        rst)))
        # cv2.imshow('effect', mask)

    def _count_fringe(self):
        if self._info[12]:
            (x1, y1), (x2, y2) = self._info[9], self._info[10]
            win = self.frame[y1:y2, x1:x2]
            if np.count_nonzero(win == self.c_blue):
                if not self._info[11]:
                    self._info[11] = True
                    self.fringe_counter += 0.5
            else: self._info[11] = False

            cv2.rectangle(self.frame, (x1, y1), (x2, y2),
                          self.c_green if not self.fringe_counter%1 else self.c_red)

    def process_img(self, img, roi = None, rsw = None):
        if rsw is None: rsw  = self._RSW
        if roi is None: roi  = self._ROI

        if rsw:
            # rsw: ymin, ymax, xmin, xmax
            cv2.imshow('src', cv2.resize(img, None, fx = 0.25, fy = 0.25))
            H, W = img.shape[:2]
            h, w = rsw[1]-rsw[0], rsw[3]-rsw[2]
            ratio = min(W/w, H/h)
            img = img[rsw[0]:rsw[1], rsw[2]:rsw[3]]
            img = cv2.resize(img, None, fx = ratio, fy = ratio)
        else: cv2.destroyWindow('src')

        if roi:
            # roi: ymin, ymax, xmin, xmax
            # here you mustn't use self.roi = img.copy()
            # because self.roi is just a pointer somewhat, which is useful
            self.roi = img[roi[0]:roi[1], roi[2]:roi[3]]
            cv2.rectangle(img,
                          (roi[2], roi[0]),
                          (roi[3], roi[1]),
                          self.c_green,
                          1, cv2.LINE_AA)
            if self._info[13]: self._update_effect_window(self.roi)
            else:              cv2.destroyWindow('effect')

        if self._info[6] and len(img.shape) > 2:
            cv2.rectangle(img,
                          self._info[0],
                          self._info[1],
                          self.c_red,
                          1, cv2.LINE_AA)
        if self._info[7] and len(img.shape) > 2:
            cv2.line(img,
                     self._info[2],
                     self._info[3],
                     self.c_blue,
                     1, cv2.LINE_AA)

        return img

    def do_ocr(self, img):
        return
        # BUG
        # if not self._FLAG_OCR: return
        # digit = img[:, i*img.shape[1]/num : (i+1)*img.shape[1]/num]
        # digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        # digit =
        # t = ''
        # digit = []
        # TODO:cascade
        # t += str(self.ocr_model.\
        #          predict_classes(cv2.resize(digit,(20,20)).reshape(1,1,20,20),
        #                          verbose = 0)[0])
        # return int(t)

    def run(self):
        while self.cam.isOpened():
            try:
                self._t = time.time()

                if self._FLAG_OCR:
                    if not int(time.time()-self.start_time)%10:
                        self.num_list.append(self.do_ocr(self.frame))

                self.frame = self.cam.read()[1]
                self.frame = self.process_img(self.frame)

                self._keyboard(cv2.waitKey(self._ftd))

                self._update_bar_window()
                self._count_fringe()

                self.real_fps = 1/(time.time()-self._t)
                self.text = 'fps: %.3f fringes: %d'%(self.real_fps, self.fringe_counter)
                cv2.putText(self.frame, self.text, self._info[8],
                            cv2.FONT_HERSHEY_DUPLEX, 0.6,
                            self.c_white, 1, cv2.LINE_AA)

                cv2.imshow(self.winname, self.frame)

            except KeyboardInterrupt:
                print('Terminated.')
                break
            except BaseException, e:
                print('{}: {}'.format(type(e), e))
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def regression(self, y, x = None):
        '''一元线性回归，如果不传入X则选取（1,2,3,4,5...）作为横坐标
        '''
        if not isinstance(y, (list, np.ndarray)):
            print('Input should be of type list or numpy array but %s found'%type(y))
            return
        if x is None: x = np.array(range(len(y)))
        y = np.array(y)
        x = np.array(x)
        xa, ya, x2a, y2a = np.average(zip(x, y, x**2, y**2), axis=0)
        xya = np.average(x * y)
        xa2, ya2 = xa**2, ya**2
        b = (xya - xa*ya) / (x2a - xa2)
        a = ya - b*xa
        r = (xya - xa*ya) / np.sqrt((x2a - xa2)*(y2a - ya2))
        result = 'x_average  : %f\n'%xa + \
                 'y_average  : %f\n'%ya + \
                 'x^2_average: %f\n'%x2a+ \
                 'y^2_average: %f\n'%y2a+ \
                 'x_average^2: %f\n'%xa2+ \
                 'y_average^2: %f\n'%ya2+ \
                 'x*y_average: %f\n'%xya+ \
                 'b          : %f\n'%b+ \
                 'a          : %f\n'%a+ \
                 'r          : %f\n'%r
        print(result)
        return xa, ya, x2a, y2a, xa2, ya2, xya, b, a, r, result

    def uncertainty(x, n = None, r = None, B_da = None, mode = 'Bezier'):
        '''本函数可以计算不确定度
        A类不确定度是对测量数据进行统计分析而获得的不确定度分量：
            1.当使用单次测量值或者n次测量值中误差最大的一次作为测量结果时
              Ua = s(x)
              s(x)为标准差（标准偏差标准误差）
              mode对应Bezier
            2.当使用n次测量值的平均值作为测量结果时
              Ua = s(x_aver) = s(x) / sqrt(n)
              n为测量次数（也有论文说n是平均值x_aver出现的次数，不过不太可信）
              mode对应Single
        B类不确定度是：
            一般使用误差限除以根三，即Ub = da/sqrt(3)
            其中da为误差限，误差限一般与测量仪器的最小分度值（最高分辨率）有关，sqrt(3)为置信区间，默认值
            1.游标卡尺的误差限为其最小分度值（i.e.1/50mm分度的误差限即为0.02mm）
            2.一般电子天平的da=0.1g
            3.石英电子停表的da=0.01s
            4.螺旋测微计和钢板尺等按最小分度的1/2计算（钢板尺0.5mm，螺旋0.005mm）
            5.电磁仪表da=a%×Nm（a%是仪表的准确度等级，Nm是量程）
            6.直流电桥da=a%×（Rx+Ro/10）其中Ro为基准值，是该量程中最大的10的整数幂
              (e.g.一个灵敏电流计的示数盘最大值是25A，选择10nA量程，则量程比率为0.4nA/div，最大读数为10nA
              10的所有整数幂中不大于10nA的最大值就是10e-08A即10nA，所以Io就是10nA，对应Ro可以计算)
            7.灵敏度误差da=0.2/S=0.2*dx/dn（其中S=dn/dx，是指当被测量x变化dx时，仪表指针变化dn格）

        Parameters
        ----------
        x : array | list | int | float
            Input collected data
        n : int
            Number of datas
            e.g. 1 for Single/Regression and 8 for 1041/1042 and 10 for 1091, etc
        B_da : float
            Number da is used for calc 'Ub' and is decided by experiment tools or data collecting methods
        r : float
            Only needed in mode Regression
        mode : str
            Method to calc Ua, default 'Bezier', can be one of Bezier/Single/Regression....

        Returns
        -------
        U  : float
        Ua : float
        Ub : float

        See Also
        --------
        https://www.baidu.com/s?wd=不确定度
        '''
        if not isinstance(x, (list, np.ndarray, int, float)):
            print('Input param "x" should be list or number or numpy.array')
            return
        else: x = np.array(x)
        xa = np.average(x)
        ua, ub, u = 0, 0, 0
        if mode == 'Regression':
            if not r or not n:
                print("Required argument 'r' or 'n' not found for mode Regression")
                return
            if x.shape: b = x[0]
            else: b = x
            ua = b * np.sqrt((1/ r**2 - 1) / (n-2))
        elif mode == 'Bezier':
            if not n: n = len(x)
            ua = np.sqrt( np.sum((x-xa)**2) / (n*(n-1)) )
        elif mode == 'Single':
            if not n: n = len(x)
            ua = np.sqrt( np.sum((x-xa)**2) / (1*(n-1)) )
        else:
            print('Param "mode" should be one of Bezier/Single/Regression... but {} found.'.format(mode))
            return

        if B_da: ub = B_da / np.sqrt(3)
        u = np.sqrt(ua**2 + ub**2)
        result = 'ua: {}\nub: {}\nu: {}'.format(ua, ub, u)
        print(result)
        return u, ua, ub, result

class _virtual_cam_class(object):
    '''This class is to improve compatibility of BPE class on both image|video source
    '''
    def __init__(self, src):
        if src[-3:] in ['jpg', 'png', 'bmp']:
            self.img = cv2.imread(src)
            self.read = lambda *args, **kwargs: (True, self.img.copy())
            self.is_open = True
        elif src[-3:] == 'gif':
            try:
                import image2gif
                src = image2gif.readGif(src)
                self.read = iter(zip(len(src)*[True], src)).next
                self.is_open = True
            except:
                self.is_open = False
        else:
            self.isOpened = False
    def isOpened(self):
        return self.is_open
    def release(self):
        self.is_open = False

if __name__ == '__main__':
#    if sys.argv[1:].__len__():
#        video_path = sys.argv[2]
#    else: video_path = None
    # video_path = 'test.avi'
    # img_test = 'Screenshot from 2017-12-13 09-08-32.png'
    p = BPE()
    p.run()
