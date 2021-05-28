import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm

#1 initialize required values
wcam,hcam=640,480
frameR=100
smoothening=7
plocx,plocy=0,0
clocx,clocy=0,0
wsrc,hsrc=autopy.screen.size()
prev_time=0
cur_time=0
cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
#2 call the module from same level directory
detector=htm.HandDetector(maxHands=1)
#3 run the camera 
while True:
	res,img=cap.read()
	#4 catch the frames
	img=detector.findHands(img)
	#5 pass the frames into detector
	lmlist,bbox=detector.findPosition(img,radius=10)
	#6 collect 21 position details of hand gesture
	cv2.rectangle(img,(frameR,frameR),(wcam-frameR,hcam-frameR),(255,0,255),2)
	if len(lmlist)!=0:
		x1,y1=lmlist[8][1:]
		x2,y2=lmlist[12][1:]
		fingers=detector.fingersUp()
		#7 collect finger tip details
		if fingers[1]==1 and fingers[2]==0:
			#8 if index is up and middle finger is down then it is in move state
			x3=np.interp(x1,(frameR,wcam-frameR),(0,wsrc))
			y3=np.interp(y1,(frameR,hcam-frameR),(0,hsrc))
			clox=plocx+(x3-plocx)/smoothening
			cloy=plocy+(y3-plocy)/smoothening
			try:
				autopy.mouse.move(wsrc-clox,cloy)
			except:
				print("ntg")
			cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
			plocx,plocy=clox,cloy
		if fingers[1]==1 and fingers[2]==1:
			#9 if index is up and middle finger is up then it is in click state
			length,img,lineinfo=detector.findDistance(8,12,img)
			if length<40:
				#10 if length between index tip and middle tip is less than 40 
				cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(255,0,0),cv2.FILLED)
				autopy.mouse.click()
	#11 calculate frame rate
	cur_time=time.time()
	fps=1/(cur_time-prev_time)
	prev_time=cur_time 
	cv2.putText(img,"Fps : "+str(int(fps)),
		        (10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
	cv2.imshow("Image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break