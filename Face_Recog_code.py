import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path='images'                                                               #PATH OF THE IMAGES FOLDER WHERE ALL IMAGES KEPT
images = []                                                                 #CREATING AN EMPTY LIST FOR ADDING IMAGES
personNames = []                             
            
             #NAME OF THE PERSON IN THE IMAGE
myList = os.listdir(path)                                                   #CREATES A LIST OF THE GIVEN PATH 
#print(myList)  

for cu_img in myList:                                                       #FOR CURRENT IMAGE IN THE LIST
    current_Img = cv2.imread(f'{path}/{cu_img}')                            #READING THE CURRENT_IMAGE    
    images.append(current_Img)                                              #ADDING THE CURRENT_IMAGE IN IMAGES
    personNames.append(os.path.splitext(cu_img)[0])                         #ADDING PERSON NAME AFTER SPLITING FROM THE IMAGE NAME
print(personNames)                                                          #PRINTS NAME OF THE PERSON IN IMAGE

def faceEncodings(images):                                                  #ENCODES THE IMAGE IN FORM OF ARRAY FOR COMPARISION
    encodeList = []                                                         #EMPTY ENCODING LIST FORMED WHERE THE ENCODINGS OF THE IMAGE CAN BE STORED
    for img in images:                                                      #SELECTING AN IMAGE FROM THE IMAGE LIST
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                          #CONVERTING THE CURRENT_IMAGE FROM BGR2RGB FORMAT
        encode = face_recognition.face_encodings(img)[0]                    #ENCODES THE CURRENT_IMAGE
        encodeList.append(encode)                                           #ADDS IT TO THE ENCODE LIST FORMED ABOVE
    return encodeList                                                       #RETURNS THE SAME FOR COMPARISION FURTHER

def attendance(name):                                                       #ATTENDANCE FUNCTION WHICH WILL MAKE A LOG FILE OF THE ATTENDEES     
    with open('Attendance.csv', 'r+') as f:                                 #THE LOG FILE WHERE ATTENDEES INFORMATION IS STORED
        myDataList = f.readlines()                                          #DATALIST THAT READS THE CONTENT OF THE LOG FILE 
        nameList = []                                                       #SPECIFIES THE NAME OF THE PERSON IDENTIFIED IN THE PHOTO
        for line in myDataList:                                             #LOOP TO SEPERATE THE ENTRIES
            entry = line.split(',')                                         #SEPERATES THE ENTRIES TO DISTINGUISH
            nameList.append(entry[0])                                       #REGISTERS THE NEW ENTRIES MADE    
        if name not in nameList:                                            #MAKES NEW ENTRY IN LOG FILE AND ENSURES THE ENTRIES DO NOT REPEAT 
            time_now = datetime.now()                                       #NOTES THE TIME WHEN ENTRY MADE
            tStr = time_now.strftime('%H:%M:%S')                            #TIME FORMAT IN WHICH ENTRY STORED
            dStr = time_now.strftime('%d/%m/%Y')                            #DATE FORMAT IN WHICH ENTRY STORED
            f.writelines(f'\n{name},{tStr},{dStr}')                         #PARAMETERS THAT ARE PRINTED IN THE LOG FILE    

encodeListKnown = faceEncodings(images)                                                  #CALLS THE FACE_ENCODING FUNCTION FOR THE IMAGES AND STORES IN ENCODE_LIST_KNOWN
print('All Encodings Complete!!!')                                                       #ASSURES ALL IMAGES ENCODED

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    
                                                     #TO TURN ON THE WEBCAM

while True:
    ret, frame = cap.read()                                                              #USED TO READ THE CAPTURED IMAGE
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)                                  #GIVES SIZE TO THE CAPTURING FRAME
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)                                       #CONVERTS THE CAPTURED FACE FROM BGR2RGB FORMAT   

    facesCurrentFrame = face_recognition.face_locations(faces)                           #USED TO RECOGNIZE THE FACE    
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)      #USED TO ENCODE THE RECOGNIZED FACE

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):              #ENCODING OF FACE AND LOCATION OF FACE GET THEIR PARAMETERS IN THE ZIPPED FORM
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)            #MATCHES (compares) THE FACE FROM THE ENCODING OF KNOWN LIST AND THE CURRENT FACE
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)            #DISTANCE OF THE FACE FROM THE CAMERS HAS BEEN SET TO CALCULATE THE ENCODING
        
        matchIndex = np.argmin(faceDis)                                                  #FINDS THE INDEX VALUE OF THE INPUT IMAGE WHEN AT RIGHT DISTANCE   
        
        if matches[matchIndex]:                                                          #IF THE INDEX FOR MATCH_INDEX EXISTS AND MATCHES. 
           name = personNames[matchIndex].upper()                                        #NAME STORES THE PERSON_NAME AT THAT INDEX.
           print(name)                                                                   #PRINTS THE NAME OF THE PERSON IDENTIFIED.
           y1, x2, y2, x1 = faceLoc                                                      #DIMENSIONS FOR THE RECTANGULAR BOX.
           y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4                               #DIMNESIONS MULTIPLIED BY 4 TO OBTAIN CORRECT SIZE   
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)                      #FORMS THE FRAME AND GIVES IT THE COLOR. THE CODE IN NEXT LINE HELPS TO NAME THE PERSON IDENTIFIED IN REAL TIME.
           cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
           attendance(name)

        else:
           y1, x2, y2, x1 = faceLoc                                                      #DIMENSIONS FOR THE RECTANGULAR BOX.
           y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4                               #DIMNESIONS MULTIPLIED BY 4 TO OBTAIN CORRECT SIZE   
           cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)                      #FORMS THE FRAME AND GIVES IT THE COLOR. THE CODE IN NEXT LINE HELPS TO NAME THE PERSON IDENTIFIED IN REAL TIME.
           cv2.putText(frame, 'UNAUTHORIZED', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) 

    cv2.imshow('camera', frame)                                                          #CAMERA OPENS AND THE FRAME IS FORMED IF FACE IDENTIFIED
    if cv2.waitKey(1) == 13:                                                             #ENTER KEY HELPS TO STOP THE CAMERA IMAGE CAPTURING
        break                                                                            #BREAKS THE PROGRAM   

cap.release()                                                                            #THIS WILL STOP CAMERA FUNCTIONING   
cv2.destroyAllWindows()                                                                  #THIS WILL END ALL THE PROCESSES TAKING PLACE   
