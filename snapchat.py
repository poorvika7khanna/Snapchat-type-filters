import cv2
import numpy as np

eyes_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("Nose18x15.xml")

cap = cv2.VideoCapture(0)

glasses = cv2.imread("glasses.png",-1)
mst = cv2.imread("mustache.png",-1)

while True:
  ret,frame = cap.read()

  if ret == False:
    continue
  eye = eyes_cascade.detectMultiScale(frame,1.3,5)[0]
  eye_x,eye_y,eye_w, eye_h= eye
  glasses=cv2.resize(glasses,(eye_w+30,eye_h+40))
  for i in range(glasses.shape[0]):
    for j in range(glasses.shape[1]):
      if (glasses[i,j,3]>0):
        frame[eye_y+i-20,eye_x+j-14, :]=glasses[i,j,:-1]
  nose = nose_cascade.detectMultiScale(frame,1.3,5)[0]
  print(nose)
  n_x,n_y,n_w, n_h= nose
  mst=cv2.resize(mst,(n_w+25,n_h-17))
  for i in range(mst.shape[0]):
    for j in range(mst.shape[1]):
      if (mst[i,j,3]>0):
        frame[n_y+i+45,n_x+j-10, :]=mst[i,j,:-1]
  cv2.imshow("Faces",frame)

  key_pressed = cv2.waitKey(1) & 0xFF
  if key_pressed == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()