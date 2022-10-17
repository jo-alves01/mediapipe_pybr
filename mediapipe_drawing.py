import cv2
import numpy as np
import hand 

drawColor=(255,255,255)#setting white color
brushThickness = 3
eraserThickness = 30

xp, yp = 0, 0
imgCanvas = np.zeros((550, 550, 3), np.uint8)# defining canvas

detector = hand.handDetector(detectionCon=0.8)#making object
cap=cv2.VideoCapture(1)

while True:

    # 1. Import image
    success, img = cap.read()
    img=cv2.flip(img,1)#for neglecting mirror inversion
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)#using functions fo connecting landmarks
    lmList,bbox = detector.findPosition(img, draw=False)#using function to find specific landmark position,draw false means no circles on landmarks
    
    if len(lmList)!=0:
        x1, y1 = lmList[8][1],lmList[8][2]# tip of index finger
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2] == False:
            drawColor = (255, 255, 255)
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)#drawing mode is represented as circle
            if xp == 0 and yp == 0:#initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                xp, yp = x1, y1 # so to avoid that we set xp=x1 and yp=y1
            #till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also
            
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1 # giving values to xp,yp everytime

        if fingers[1] and fingers[2] == True:
            drawColor = (0, 0, 0)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            xp,yp=x1,y1

    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Raw Webcam Feed', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break