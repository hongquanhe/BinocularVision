import os
import time
import cv2
import numpy as np

def merge_video():
    path = 'img/output' # 图片序列所在目录，文件名：0.jpg 1.jpg ...
    dst_path = 'img/run.avi' # 生成的视频路径

    filelist = os.listdir(path)
    filepref = [os.path.splitext(f)[0] for f in filelist]
    filepref.sort(key = int) # 按数字文件名排序
    filelist = [f + '.jpg' for f in filepref]
    # filepref.sort(key = str) # 按数字文件名排序
    # filelist = [f + '.bmp' for f in filepref]

    width = 200
    height = 200
    fps = 10
    col_cnt = 2 #显示视频时的图片列数(倍数)
    vw = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width * col_cnt, height))

    for file in filelist:
        if file.endswith('.jpg'):
        # if file.endswith('.bmp'):
            file = os.path.join(path, file)
            img = cv2.imread(file)
            img = np.hstack((img, img))  # 如果并排两列显示
            vw.write(img)

    vw.release()


if __name__ == "__main__":
    merge_video()
    print('end')
