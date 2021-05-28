import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

prev_time=0
cur_time=0
cap=cv2.VideoCapture(0)
detector=htm.HandDetector()
while True:
	res,img=cap.read()
	img=detector.findHands(img)
	#lmlist=detector.findPosition(img)
	detector.findSpecId(img,0,3,True,16)
	cur_time=time.time()
	fps=1/(cur_time-prev_time)
	prev_time=cur_time
	cv2.putText(img,"Fps : "+str(int(fps)),
		        (10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
	cv2.imshow("Image",img) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break