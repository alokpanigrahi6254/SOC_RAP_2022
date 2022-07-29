import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('Alarm.mpeg')

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
#cascade classifier uses sliding window approach starting from minsize, slides through image resizes and again slides and capture true outputs boxes.

lbl=['Close','Open']

model = load_model('eyeclassifier.h5')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
rpred=0
lpred=0

while(True):
    ret, frame = cap.read() #ret=True if frame is received
    height,width = frame.shape[:2]  #img.shape gives tuple of rows, columns and channels

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect multiscale works only gray images
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    #scaleFactor: Parameter specifying how much the image size is reduced at each image scale, a 10%reduction in size in this case.
    #minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    #minSize: Minimum possible object size. Objects smaller than that are ignored.

    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    for (x,y,w,h) in faces:
    #x,y coordinates of top left corner and w and h width and height of the rectangle
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 ) #making rectangle around the face detected
        #(x,y): Top left corner of rect, ie start
        #(x+w,y+h): Bottom right corner of rect, ie end
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24)) #resizing of image into (24,24) pixels
        r_eye= r_eye/255 #normalizing each pixels
        r_eye=  r_eye.reshape(24,24,-1) #making it a 3D array having 24 rows, having 24 columns, having 1 element (-1 adjusts elements in last dimension automatically)
        #print(r_eye)
        r_eye = np.expand_dims(r_eye,axis=0) 
        rpred = model.predict(r_eye)[0][0] #result is 2D array of result reqd for each image in a batch
        
        print(rpred)        
        if(rpred>=0.85):
            lbl='Closed' 
        if(rpred<=0.20):
            lbl='Open'
        break #detect only the one(right eye) in one frame

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)[0][0]
        if(lpred>=0.85):
            lbl='Closed'   
        if(lpred<=0.20):
            lbl='Opend'
        break #detect only the one(left eye) in one frame

    if(rpred>=0.80 and lpred>=0.80):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>5):
        try:
            sound.play()
        except:  
            pass
        #The try block is used to check some code for errors i.e the code inside the try block will execute when there is no error in the program. 
        #Whereas the code inside the except block will execute whenever the program encounters some error in the preceding try block.
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) #making red rectangle on the complete frame, in case of error.
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
