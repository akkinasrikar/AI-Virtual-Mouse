import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils

prev_time=0
cur_time=0

while True:
	res,img=cap.read()
	imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	results=hands.process(imgRGB)
	if results.multi_hand_landmarks:
		for h in results.multi_hand_landmarks:
			for Id,Lm in enumerate(h.landmark):
				height,width,c=img.shape
				cx,cy=int(Lm.x*width),int(Lm.y*height)
				if Id==4:
					cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
			mpdraw.draw_landmarks(img,h,mphands.HAND_CONNECTIONS)

	cur_time=time.time()
	fps=1/(cur_time-prev_time)
	prev_time=cur_time

	cv2.putText(img,"Fps : "+str(int(fps)),
		        (10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

	cv2.imshow("Image",img) 
	cv2.waitKey(1)
