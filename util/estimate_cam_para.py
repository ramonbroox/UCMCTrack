import argparse
import os
import cv2
import math
import numpy as np
import copy


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

class CameraPara:

    def open(self,file):

        # 读取文件
        with open(file, 'r') as f:
            data = f.readlines()

        # 从文件中读取旋转矩阵
        R = []
        for i in range(1, 4):
            row = data[i].split()
            R.append([float(x) for x in row])
        R = np.array(R)

        # 从文件中读取平移向量
        T = []
        for i in range(6, 7):
            row = data[i].split()
            T.append([float(x)/1000.0 for x in row])
        T = np.array(T)

        self.Ko = np.column_stack((R, T.T))
        self.Ko = np.row_stack((self.Ko, np.array([0,0,0,1])))

        # 从文件中读取内参矩阵
        Ki = []
        for i in range(9, 12):
            row = data[i].split()
            Ki.append([float(x) for x in row])
        self.Ki =  np.array(Ki)
        # 在self.Ki中添加一列[0,0,0]
        self.Ki = np.column_stack((self.Ki, np.array([0,0,0])))

    @property
    def focal(self):
        return float(self.Ki[0,0])

    @property
    def T(self):
        return float(self.Ki[0,2]) , float(self.Ki[1,2]) , float(self.Ko[2,3])

    @property
    def r_angles_deg(self):
        return 0,0,0
       #return tuple((180.0/np.pi)*rotationMatrixToEulerAngles(self.Ko[0:3,0:3]))



    def xy2uv(self,x,y,z=0):
        # 计算uv
        uv = np.dot(self.Ki, np.dot(self.Ko, np.array([x,y,z,1])))
        # 归一化
        uv = uv/uv[2]
        return int(uv[0]), int(uv[1])


class MappedTrackbar:
    def __init__(self, name, window, min_v, max_v, v0, glob_var, after_chg_callback=None):
        self._min_v = min_v
        self._max_v = max_v
        self._glob_var = glob_var
        self._after_chg_callback = after_chg_callback
        def_v0 = int(2147483640*(v0 - min_v)/(max_v - min_v))
        if def_v0 < 0:
            def_v0 = 0
        print(name, window, def_v0, self._map_val)
        cv2.createTrackbar(name, window, def_v0, 2147483647, self._map_val)

    def _map_val(self, v):
        new_v = self._min_v + (self._max_v-self._min_v)*(v/2147483647.0)
        globals()[self._glob_var] = new_v
        if self._after_chg_callback is not None:
            self._after_chg_callback(new_v)

def xy2uv(x,y,Ki,Ko, z=0):
    # 计算uv
    uv = np.dot(Ki, np.dot(Ko, np.array([x,y,z,1])))
    # 归一化
    uv = uv/uv[2]
    return int(uv[0]), int(uv[1])


# 定义滑动条回调函数

def update_value_display():
    global value_display,g_theta_x,g_theta_y,g_theta_z,g_focal,g_tz,g_ty,g_tz
    value_display.fill(0)  # 清空图像
    text = f"theta_x: {g_theta_x:.2f}"
    cv2.putText(value_display, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"theta_y: {g_theta_y:.2f}"
    cv2.putText(value_display, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"theta_z: {g_theta_z:.2f}"
    cv2.putText(value_display, text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"focal: {g_focal}"
    cv2.putText(value_display, text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"Tz: {g_tz:.2f}"
    cv2.putText(value_display, text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"Tx: {g_tx:.2f}"
    cv2.putText(value_display, text, (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"Ty: {g_ty:.2f}"
    cv2.putText(value_display, text, (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow('Values', value_display)


cv2.namedWindow('Values')
# 初始化一个空白图像来显示实际值
value_display = np.zeros((400, 300), dtype=np.uint8)
g_theta_x = 0
g_theta_y = 0
g_theta_z = 0
g_focal = 0
g_tx = 0
g_ty = 0
g_tz = 0

orimg = None

def main(args):


    if args.img.endswith(".mp4"):
        cam = cv2.VideoCapture(args.img)
        cam.grab()
        _ , orimg = cam.read()
    else:
        orimg = cv2.imread(args.img)

    # 获取img的大小
    img = orimg.copy()
    height, width = img.shape[:2]

    if not os.path.exists(args.cam_para):
        with open(args.cam_para,"w") as f:
            halfw = int(width/2)
            halfh = int(height/2)
            f.write("RotationMatrices\n")
            f.write("-0.37 -0.92 -0.057\n")
            f.write("-0.51 0.25 -0.81\n")
            f.write("0.77 -0.27 -0.57\n")
            f.write("\n")
            f.write("TranslationVectors\n")
            f.write("0 1000 35000\n")
            f.write("\n")
            f.write("IntrinsicMatrix\n")
            f.write(f"1000 0 {halfw}\n")
            f.write(f"0 1000 {halfh}\n")
            f.write("0 0 1\n")
            f.write("\n")

    global g_theta_x,g_theta_y,g_theta_z,g_focal,g_tz,g_tx,g_ty

    cam_para = CameraPara()
    cam_para.open(args.cam_para)
    g_theta_x, g_theta_y, g_theta_z = cam_para.r_angles_deg
    g_tx, g_ty, g_tz = cam_para.T
    g_focal = cam_para.focal

    cv2.namedWindow('CamParaSettings')
    # 添加ui界面来修改theta_x,theta_y,theta_z, 调节访问是-10到10，间隔0.2

    theta_x_bar = MappedTrackbar('theta_x', 'CamParaSettings', -180, 180.0, g_theta_x, 'g_theta_x', after_chg_callback=lambda x: update_value_display())
    theta_y_bar = MappedTrackbar('theta_y', 'CamParaSettings', -180, 180.0, g_theta_y, 'g_theta_y', after_chg_callback=lambda x: update_value_display())
    theta_z_bar = MappedTrackbar('theta_z', 'CamParaSettings', -180, 180.0, g_theta_z, 'g_theta_z', after_chg_callback=lambda x: update_value_display())
    focal_bar = MappedTrackbar('focal', 'CamParaSettings', -3500, 3500, g_focal, 'g_focal', after_chg_callback=lambda x: update_value_display())
    tx_bar = MappedTrackbar('Tx', 'CamParaSettings', -width, width, g_tx, 'g_tx', after_chg_callback=lambda x: update_value_display())
    ty_bar = MappedTrackbar('Ty', 'CamParaSettings', -height, height, g_ty, 'g_ty', after_chg_callback=lambda x: update_value_display())
    tz_bar = MappedTrackbar('Tz', 'CamParaSettings', -100, 100, g_tz, 'g_tz', after_chg_callback=lambda x: update_value_display())

    update_value_display()
    # 循环一直到按下q键
    write_mod = False
    while True:
        theta_x = g_theta_x/180.0*np.pi
        theta_y = g_theta_y/180.0*np.pi
        theta_z = g_theta_z/180.0*np.pi
        Ki = copy.copy(cam_para.Ki)
        Ko = copy.copy(cam_para.Ko)
        R = Ko[0:3,0:3]
        Rx = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y),0,np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
        R = np.dot(R, np.dot(Rx, np.dot(Ry,Rz)))
        Ko[0:3,0:3] = R
        Ko[2,3] = g_tz
        Ki[0,0] = g_focal
        Ki[1,1] = g_focal
        Ki[0,2] = g_tx
        Ki[1,2] = g_ty
        img = orimg.copy()

        # x取值范围0-10，间隔0.1
        for x in np.arange(-10,10,1):
            for y in np.arange(-10,10,1):
                u,v = xy2uv(x,y,Ki,Ko)
                r = 3
                if x == 0 and y == 0:
                    r = 8
                cv2.circle(img, (u,v), r, (int((x+10)*100), int((y+10)*100) ,int((y+10)*10)), -1)

        zerozero = xy2uv(0,0,Ki,Ko)
        zerotop = xy2uv(0,0,Ki,Ko, z=180)
        cv2.line(img, zerozero, zerotop, (255,0,0), 3)

        # 修改img的大小
        #img = cv2.resize(img, (int(width*0.5),int(height*0.5)))
        cv2.imshow('img', img)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break
        if key == ord('w'):
            write_mod = True
            break

    if write_mod:
        with open(args.cam_para, 'w') as f:
            f.write("RotationMatrices\n")
            for i in range(3):
                for j in range(3):
                    f.write(str(R[i,j])+" ")
                f.write("\n")
            f.write("\nTranslationVectors\n")
            for i in range(3):
                f.write(str(int(Ko[i,3]*1000))+" ")
            f.write("\n\nIntrinsicMatrix\n")
            for i in range(3):
                for j in range(3):
                    f.write(str(int(Ki[i,j]))+" ")
                f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--img', type=str, required= True,help='The image file ')
    parser.add_argument('--cam_para', type=str, required= True,help='The estimated camera parameters file ')
    args = parser.parse_args()
    main(args)
