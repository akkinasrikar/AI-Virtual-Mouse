import cv2
import mediapipe as mp
import time,math

class HandDetector:
	def __init__(self,Mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
		self.mode=Mode
		self.maxHands=maxHands
		self.detectionCon=detectionCon
		self.trackCon=trackCon
		self.mphands=mp.solutions.hands
		self.hands=self.mphands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
		self.mpdraw=mp.solutions.drawing_utils
		self.tipIds = [4, 8, 12, 16, 20]

	def findHands(self,img,draw=True):
		imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		self.results=self.hands.process(imgRGB)
		if self.results.multi_hand_landmarks:
			for h in self.results.multi_hand_landmarks:
				if draw==True:
					self.mpdraw.draw_landmarks(img,h,self.mphands.HAND_CONNECTIONS)
		return img

	def findPosition(self,img,handNo=0,draw=True,radius=15):
		self.lmList=[]
		xList = []
		yList = []
		bbox = []
		if self.results.multi_hand_landmarks:
			myhand=self.results.multi_hand_landmarks[handNo]
			for Id,Lm in enumerate(myhand.landmark):
				height,width,c=img.shape
				cx,cy=int(Lm.x*width),int(Lm.y*height)
				xList.append(cx)
				yList.append(cy)
				self.lmList.append([Id,cx,cy])
				if draw:
					cv2.circle(img,(cx,cy),radius,(255,0,255),cv2.FILLED)
			xmin, xmax = min(xList), max(xList)
			ymin, ymax = min(yList), max(yList)
			bbox = xmin, ymin, xmax, ymax
			if draw:
				cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)
		return self.lmList,bbox
	def fingersUp(self):
		fingers = []
		if self.lmList[self.tipIds[0]][1]>self.lmList[self.tipIds[0]-1][1]:
			fingers.append(1)
		else:
			fingers.append(0)
		for id in range(1, 5):
			if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
				fingers.append(1)
			else:
				fingers.append(0)
		return fingers
	def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
		x1, y1 = self.lmList[p1][1:]
		x2, y2 = self.lmList[p2][1:]
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		if draw:
			cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
			cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
			cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
			cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
			length = math.hypot(x2 - x1, y2 - y1)
		return length, img, [x1, y1, x2, y2, cx, cy]

	def findSpecId(self,img,handNo=0,ID="all",draw=True,radius=15):
		lmList=[]
		if self.results.multi_hand_landmarks:
			myhand=self.results.multi_hand_landmarks[handNo]
			for Id,Lm in enumerate(myhand.landmark):
					height,width,c=img.shape
					cx,cy=int(Lm.x*width),int(Lm.y*height)
					lmList.append([Id,cx,cy])
					if draw and Id==ID:
						cv2.circle(img,(cx,cy),radius,(255,0,255),cv2.FILLED)
		return lmList

