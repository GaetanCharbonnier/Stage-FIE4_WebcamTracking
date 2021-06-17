# -*- coding: utf-8 -*-
"""
@author: Charbonnier Gaetan 
"""

import math
import cv2 as cv
import argparse
import time
from datetime import datetime
from time import strftime
from datetime import timedelta
import csv
import math
import numpy as np
from imutils import face_utils
import dlib

# distance from camera to face measured
Known_distance = 65  # cm
# measure your face width
Known_width = 14    # cm

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
PURPLE = (128, 0, 128)
PINK = (147, 20, 255)

# time Start
start_time = time.time()
# Date Start
DateStart = datetime.now()
DateStart.strftime("%y %m %d")
print('\n',"Start camera : ", DateStart.strftime("%y-%m-%d"))
# Time Start 
startDate = datetime.now()
startDate.strftime("%H:%M:%S")
print('\n',"Start camera : ", startDate.strftime("%H:%M:%S"))

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "data/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Camera Object
capVid = cv.VideoCapture(0, cv.CAP_DSHOW)  # Number According to your Camera
if(capVid.isOpened() == False): 
    print("Error: the resource is busy or unvailable")
else:
        print("The video source has been opened correctly")
Distance_level = 0

# Create VideoWriter object
# VideoWriter (const String &filename, int fourcc, double fps, Size frameSize)
out = cv.VideoWriter(
    'VideOutput.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

# face/eyes detector object
parser = argparse.ArgumentParser(description='Code Cascade Classifier')
parser.add_argument('--face_detector', help='Path to face detector.',
                    default='data/haarcascade/haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_detector', help='Path to eyes detector.',
                    default='data/haarcascade/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument(
    '--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_detector_name = args.face_detector
eyes_detector_name = args.eyes_detector

face_detector = cv.CascadeClassifier(
    "data/haarcascade/haarcascade_frontalface_default.xml")
eyes_detector = cv.CascadeClassifier(
    "data/haarcascade/haarcascade_eye.xml")
# Load the cascades
if not face_detector.load(cv.samples.findFile(face_detector_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_detector.load(cv.samples.findFile(eyes_detector_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera

# focal length finder function
def FocalLength(measured_distance, real_width, width_in_rf_image):
    # Function Discrption (Doc String)
    '''
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), by using:
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
        Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
        Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
    '''
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    '''
    This Function Estimates the distance between object and camera using arguments :
        Focal_length(float): return by the Focal_Length_Finder function
        Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    '''
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance
            
# face detection Fauction
def face_data(image, CallOut, Distance_level):
    '''
    This function Detect face and Draw Rectangle and display the distance over Screen
        Image(Mat): simply the frame
        Call_Out(bool): If want show Distance and Rectangle on the Screen or not
        Distance_Level(int): which change the line according the Distance changes(Intractivate)

    return  face_width(int): it is width of face in the frame which allow us to calculate the distance and find focal length
    return face(list): length of face and (face paramters)
    return face_center_x: face centroid_x coordinate(x)
    return face_center_y: face centroid_y coordinate(y)
    '''

    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        line_thickness = 2
        # print(len(faces))
        LLV = int(h*0.12)
        # print(LLV)

        # cv.rectangle(image, (x, y), (x+w, y+h), BLACK, 1)
        cv.line(image, (x, y+LLV), (x+w, y+LLV), (GREEN), line_thickness)
        cv.line(image, (x, y+h), (x+w, y+h), (GREEN), line_thickness)
        cv.line(image, (x, y+LLV), (x, y+LLV+LLV), (GREEN), line_thickness)
        cv.line(image, (x+w, y+LLV), (x+w, y+LLV+LLV), (GREEN), line_thickness)
        cv.line(image, (x, y+h), (x, y+h-LLV), (GREEN), line_thickness)
        cv.line(image, (x+w, y+h), (x+w, y+h-LLV), (GREEN), line_thickness)

        img = cv.rectangle(image, (x, y), (x+w, y+h), (GREEN), 1)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (PURPLE), 1)

        print ('x :' +str(x), 'y :'+str(y), 'x+w :' +str(x+w), 'y+h :' +str(y+h))

        face_width = w
        # Drawing circle at the center of the face
        face_center_x = int(w/2)+x
        face_center_y = int(h/2)+y
        print('face_center_x :', face_center_x, 'face_center_y :', face_center_y)

        center = (face_center_x,face_center_y)
        print("Center of Rectangle is :", center)

        if CallOut == True:
            # cv.line(image, (x,y), (face_center_x, face_center_y), COLOR,1)
            cv.line(image, (x, y-11), (x+180, y-11), (PINK), 28)
            cv.line(image, (x, y-11), (x+180, y-11), (CYAN), 20)

    return face_width, faces, face_center_x, face_center_y

# reading reference image from directory
ref_image = cv.imread("Ref_image.png")

ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print("Focal lenght found", Focal_length_found)

liste=[]
currentFrame = 0
while True:
    # load the input image and convert it to grayscale
    _, frame = capVid.read()
    # calling face_data function
    # Distance_level =0

    # Handle the mirroring of the current frame
    frame = cv.flip(frame,1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        '''
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them (68 landmarks) on the image
        for (x, y) in shape:
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
        '''

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(
        frame, True, Distance_level)
    # finding the distance by calling function Distance finder
    for (face_x, face_y, face_w, face_h) in Faces:
    
        if face_width_in_frame != 0:

            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            # Drawing Text on the screen
            Distance_level = int(Distance)
            if Distance_level < 20:
                Distance_level = 20 # la camera ne detectera pas de visage en dessous de 20 cm
            time.sleep(0.5)
            print ('Distance : ', Distance_level)
            liste.append(Distance_level)
            print (liste,'\n')
            cv.putText(frame, f"Distance {Distance_level} cm",(face_x, face_y-6), cv.FONT_HERSHEY_COMPLEX, 0.5, (BLACK), 2)
            #cv.putText(frame, f"Pitch {pitch} cm",(face_x-6, face_y), cv.FONT_HERSHEY_COMPLEX, 0.5, (BLACK), 2)
            #cv.putText(frame, f"Roll {roll} cm",(face_x-6, face_y-6), cv.FONT_HERSHEY_COMPLEX, 0.5, (BLACK), 2)
            #cv.putText(frame, f"Yaw {yaw} cm",(face_x-6, face_y-6), cv.FONT_HERSHEY_COMPLEX, 0.5, (BLACK), 2)

    cv.imshow("frame", frame)
    out.write(frame)
    if cv.waitKey(1) == ord("q"): 
        Distance_AVG=sum(liste)/len(liste)
        print("moyenne de distance :", "{0:.2f}".format(Distance_AVG))
        break                 
    currentFrame += 1

# Datetime end
DateEnd = datetime.now()
DateEnd.strftime("%y-%m-%d")
print("End camera :", DateEnd.strftime("%y-%m-%d"))
# Datetime end
endDate = datetime.now()
endDate.strftime("%H:%M:%S")
print("End camera :", endDate.strftime("%H:%M:%S"))

# Enlapsed time (s)
end_time = time.time()
Enlapsedtime = (end_time - start_time)
print("Temps actif: %0.2f" % Enlapsedtime, "secondes")
# Enlapsed time format hh:mm:ss
deltatime = str(timedelta(seconds=Enlapsedtime))
print("Temps actif: ", deltatime,'\n')

'''
# creation file csv
with open('DistanceData.csv','w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'Distance_AVG', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'DateStart' : DateStart.strftime("%y-%m-%d"),'Start_camera' : startDate.strftime("%H:%M:%S"), 'Distance_AVG' : "{0:.2f}".format(Distance_AVG), 
    'DateEnd' : DateEnd.strftime("%y-%m-%d"),'End_camera' : endDate.strftime("%H:%M:%S"), 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})
'''

# Write line csv
with open('DistanceData.csv','a', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'Distance_AVG', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writerow({'DateStart' : DateStart.strftime("%y-%m-%d"),'Start_camera' : startDate.strftime("%H:%M:%S"), 'Distance_AVG' : "{0:.2f}".format(Distance_AVG), 
    'DateEnd' : DateEnd.strftime("%y-%m-%d"),'End_camera' : endDate.strftime("%H:%M:%S"), 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})

capVid.release()
# out.release()
cv.destroyAllWindows()