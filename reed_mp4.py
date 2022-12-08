import cv2
vc = cv2.VideoCapture('video/test.mp4')
c=0
rval = vc.isOpened()

while rval:
    c = c + 1
    rval, frame = vc.read()
    if rval:
        cv2.imwrite('./video/'+str(c) + '.jpg', frame) #命名方式
        print(c)
    else:
        break
vc.release()
