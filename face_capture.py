#!/usr/bin/python3

import cv2
import os
import uuid

face_cascade = cv2 .CascadeClassifier('./haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
captures_count = 100
face_uuid = str(uuid.uuid4())
out_dir = f'./output/face/{face_uuid}/'

cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


def create_directory():
    os.mkdir(out_dir)
    print(f'\n [Info] Created directory {out_dir}\n')

def open_camera():
    count = 1
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            file_name = f'{out_dir}{count}.png'
            cv2.imwrite(file_name, gray[y:y+h, x:x+w])
            print(f' [{count}] Capture saved into {out_dir}')
            count += 1
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        elif count >= captures_count:
            break
    print('\n [INFO] Exit \n')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_directory()
    open_camera()
