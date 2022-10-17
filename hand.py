import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):#constructor
        self.mode=mode 
        self.maxHands=maxHands
        self.modelComplex = modelComplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands#initializing hands module for the instance
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon) #object for Hands for a particular instance
        self.mpDraw=mp.solutions.drawing_utils#object for Drawing
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#converting to RGB bcoz hand recognition works only on RGB image
        self.results=self.hands.process(imgRGB)#processing the RGB image 
        if self.results.multi_hand_landmarks:# gives x,y,z of every landmark or if no hand than NONE
            for handLms in self.results.multi_hand_landmarks:#each hand landmarks in results
                if draw:
                     self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)#joining points on our hand
        
        return img

    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist=[]
        if self.results.multi_hand_landmarks:# gives x,y,z of every landmark    
            myHand=self.results.multi_hand_landmarks[handNo]#Gives result for particular hand 
            for id,lm in enumerate(myHand.landmark):#gives id and lm(x,y,z)
                h,w,c=img.shape#getting h,w for converting decimals x,y into pixels 
                px,py=int(lm.x*w),int(lm.y*h)# pixels coordinates for landmarks
                xList.append(px)
                yList.append(py)
                self.lmlist.append([id,px,py])
                if draw:
                    cv2.circle(img,(px,py),5,(255,255,255),cv2.FILLED)    
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax

        return self.lmlist,bbox

    def fingersUp(self):#checking which finger is open 
        fingers = []#storing final result
        # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:#checking x position of 4 is in right to x position of 3
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):#checking tip point is below tippoint-2 (only in Y direction)
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers