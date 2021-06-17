# -*- coding: utf-8 -*-
"""@author: Charbonnier Gaetan """
from imutils import face_utils
from datetime import datetime
# from time import strftime
from datetime import timedelta
from imutils import face_utils
import math
import cv2 as cv
import argparse
import time
import csv
import math
import numpy as np
import dlib
from numpy.lib.twodim_base import eye

Known_distance = 65  # distance from camera to face measured (cm)
Known_width = 14     # measure your face width (cm)

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

BegTime = datetime.now().time()
print('BegTime : ', BegTime)
print("hour =", BegTime.hour)
print("minute =", BegTime.minute)
print("second =", BegTime.second)
print("microsecond =", BegTime.microsecond)
print("type(BegTime) =", type(BegTime))
	
import datetime
BegDate = datetime.date.today()
print("BegDate : ", BegDate)
print("Current year:", BegDate.year)
print("Current month:", BegDate.month)
print("Current day:", BegDate.day)
print("type(BegDate) =", type(BegDate))	

'''
# Date Start
DateStart = datetime.now()
print("Start camera : ", DateStart.strftime("%y-%m-%d"))

# Time Start 
startDate = datetime.now()
startDate.strftime("%H:%M:%S")
print("Start camera : ", startDate.strftime("%H:%M:%S"),'\n')
'''

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
p = "data/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
POINTS_NUM_LANDMARK = 68

# initialize the video stream and sleep for a bit, allowing the camera sensor to warm up
print("[INFO] camera sensor warming up...")

# Camera Object :
# Playing video from Webcam
capVid = cv.VideoCapture(0, cv.CAP_DSHOW)  # Number According to your Camera, DirectShow (via videoInput) 
time.sleep(2.0)
if(capVid.isOpened() == False): 
    print("Error: the resource is busy or unvailable")
else:
    print("The video source has been opened correctly")

capVid.set(cv.CAP_PROP_FRAME_WIDTH, 800)
capVid.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

Distance_level = 0

# Create VideoWriter object :
# VideoWriter (const String &filename, int fourcc, double fps, Size frameSize)
out = cv.VideoWriter('VideOutput.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

# face/eyes detector object
parser = argparse.ArgumentParser(description='Code Cascade Classifier')
parser.add_argument('--face_detector', help='Path to face detector.', default='data/haarcascade/haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_detector', help='Path to eyes detector.', default='data/haarcascade/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_detector_name = args.face_detector
eyes_detector_name = args.eyes_detector

face_detector = cv.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")
eyes_detector = cv.CascadeClassifier("data/haarcascade/haarcascade_eye.xml")
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
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length    
    
'''   This Function Calculate the Focal Length(distance between lens to CMOS sensor), by using:
        MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
        Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
        Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
'''   
    
# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance
'''   This Function Estimates the distance between object and camera using arguments : 
        Focal_length(float): return by the Focal_Length_Finder function
        Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
'''

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


        # eyes detector (can be use later)
        #img = cv.rectangle(image, (x, y), (x+w, y+h), (GREEN), 1)
        #roi_gray = gray_image[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        #eyes = eyes_detector.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
            #cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (PURPLE), 1)



        print ('x :' +str(x), 'y :'+str(y), 'x+w :' +str(x+w), 'y+h :' +str(y+h))

        face_width = w
        # Drawing circle at the center of the face
        face_center_x = int(w/2)+x
        face_center_y = int(h/2)+y
        print('face_center_x :', face_center_x, 'face_center_y :', face_center_y)

        center = (face_center_x,face_center_y)
        print("Center of Rectangle is :", center)

        if CallOut == True:
            cv.line(image, (x, y-11), (x+180, y-11), (PINK), 28)
            cv.line(image, (x, y-11), (x+180, y-11), (CYAN), 20)

    return face_width, faces, face_center_x, face_center_y

# reading reference image from directory
ref_image = cv.imread("Ref_image.png")
ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print("Focal lenght found", Focal_length_found)

listeTime=[]
listeDist=[]
listePitch=[]
listeYaw=[]
listeRoll=[]

def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]
    
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index

def get_image_points_from_landmark_shape(landmark_shape):
    #2D image points. If you change the image, you need to change vector
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None
    image_points = np.array([
                                (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),       # Chin
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                            ], dtype="double")

    return 0, image_points

def get_image_points(img):                  
    dets = detector( img, 0 )
    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]
    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)

def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corner
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
    '''
    # Camera internals
    fy = img_size [0] # fx, longitud total de la unidad de píxeles fy
    fx = img_size[1]
    center = (fx / 2, fy / 2) # define center image
    cx = center[0]
    cy = center[1]
         # Define matrix camera:
    camera_matrix = np.array(
        [[fx, 0, cx],
         [0, fx, cy],
         [0, 0, 1]], dtype=np.float64)
    ''' 
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length,0,center[0]],
                            [0, focal_length, center[1]],
                            [0,0,1]], dtype="double"
                            )

    print("Camera Matrix :\n {}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE )
 
    print("Rotation Vector:\n {}".format(rotation_vector), "\n Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

def get_euler_angle(rotation_vector):
    # Calculate rotation angles
    rvec_matrix = cv.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv.decomposeProjectionMatrix(proj_matrix)[6]
   
    pitch2, yaw2, roll2 = [math.radians(_) for _ in eulerAngles]

    pitch2 = math.degrees(math.asin(math.sin(pitch2)))
    Y = math.degrees(math.asin(math.sin(pitch2)))
    Y = round (Y, 2) 

    roll2 = -math.degrees(math.asin(math.sin(roll2)))
    yaw2 = math.degrees(math.asin(math.sin(yaw2)))
    print('pitch2Sol: ', pitch2)
    print('pitch2Sol: ', roll2)
    print('pitch2Sol: ', yaw2)

    # Calculate rotation angles
    theta = cv.norm(rotation_vector, cv.NORM_L2)
    
    # Transformed to quaterniond
    q0 = math.cos(theta / 2)
    q1 = math.sin(theta / 2)*rotation_vector[0][0] / theta
    q2 = math.sin(theta / 2)*rotation_vector[1][0] / theta
    q3 = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    # pitch (x-axis rotation)
    t0 = 2.0 * ((q0 * q1) + (q2 * q3))
    t1 = 1.0 - 2.0 * ((q1 * q1) + (q2 * q2))
    print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    print('pitch : ', pitch)
 
    # yaw (y-axis rotation)
    t2 = 2.0 * ((q0 * q2) - (q3 * q1))
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    print('yaw : ', yaw)

    # roll (z-axis rotation)
    t3 = 2.0 * ((q0 * q3) + (q1 * q2))
    t4 = 1.0 - 2.0 * ((q2 * q2) + (q3 * q3))
    print('t3:{}, t4:{}'.format(t3, t4))
    roll = math.atan2(t3, t4)
    print('roll : ', roll)
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    return 0, Y, X, Z

def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None
    
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None
    
        ret, pitch2, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch2, yaw, roll)
        print(euler_angle_str)
        return 0, pitch2, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None

currentFrame = 0
while True:
    # load the input image and convert it to grayscale
    # Read picture. ret === True on success
    ret, frame = capVid.read()

    # Handle the mirroring of the current frame
    frame = cv.flip(frame,1)

    # Our operation on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv.putText(frame, text, (500, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (GREEN), 2)
    else:
        text2 = "0 face(s) found".format(len(rects))
        cv.putText(frame, text2, (500, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (RED), 2)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        '''
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them (68 landmarks) on the image
        for (x, y) in shape:
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
        '''
        for (i,(x, y)) in enumerate(shape):
    # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
            if i == 33:
    #something to our key landmarks _ write on frame in Green
  # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[0] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
    # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[1] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
    # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[2] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
    # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[3] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
    # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[4] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
    # save to our new key point list (keypoints = [(i,(x,y)))
    #            image_points[5] = np.array([x,y],dtype='double')
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
    #everything to all other landmarks _ write on frame in Red
                cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
                #cv.putText(frame, str(i + 1), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        size = frame.shape
        ret, image_points = get_image_points(frame)
        if ret != 0:
            print('get_image_points failed')
            continue
        
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            continue
        used_time = time.time() - start_time
        print("used_time:{} sec".format(round(used_time, 3)))
        
        ret, pitch2, yaw, roll = get_euler_angle(rotation_vector)
        euler_angle_str = 'Pitch(Y): {}, Yaw(X): {}, Roll(Z): {}'.format(pitch2, yaw, roll)
        print(euler_angle_str)

        from datetime import datetime 
        timenow = datetime.now().time()
        print('TimeNow Hour : ', timenow.hour)
        print('TimeNow : ', timenow)

        # Project a 3D point (0, 0, 1000.0) onto the image plane (We use this to draw a line sticking out of the nose)
        (nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         
        for p in image_points:
            cv.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv.line(frame, p1, p2, (255, 0, 0), 2)  # RED

        #cv.putText( frame, str(rotation_vector), (0, 100), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 )
        cv.putText( frame, euler_angle_str, (20, 20), cv.FONT_HERSHEY_PLAIN, 1, (GREEN), 1 )
        time.sleep(0.3)
        listeTime.append(timenow)
        listePitch.append(pitch2)
        listeYaw.append(yaw)
        listeRoll.append(roll)
        print ('Liste Time :', listeTime)
        print ('Liste Pitch :', listePitch)
        #print ('Liste Yaw :', listeYaw)
        #print ('Liste Roll :', listeRoll)

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)
    # finding the distance by calling function Distance finder
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            # Drawing Text on the screen
            Distance_level = int(Distance)
            if Distance_level < 20:
                Distance_level = 20 # la camera ne detectera pas de visage en dessous de 20 cm
        '''
        if face_width_in_frame == 0:  
            Distance = 0
            Distance_level = int(Distance)
        '''
        time.sleep(0.3)
        print ('Distance : ', Distance_level)
        listeDist.append(Distance_level)
        print (listeDist,'\n')
        cv.putText(frame, f"Distance {Distance_level} cm",(face_x, face_y-6), cv.FONT_HERSHEY_COMPLEX, 0.5, (BLACK), 2)
    
    cv.imshow("Output", frame)
    out.write(frame)
    if cv.waitKey(1) == 27: # 27 = Echap
    #if cv.waitKey(1) == ord("q"): 
        Distance_AVG=sum(listeDist)/len(listeDist)
        print("moyenne de distance :", "{0:.2f}".format(Distance_AVG))        
        Pitch_AVG=sum(listePitch)/len(listePitch)
        print("moyenne de Pitch :", Pitch_AVG)
        break
    currentFrame += 1

import datetime
EndTime = datetime.date.today()
print("EndTime : ", EndTime)
print("Current year:", EndTime.year)
print("Current month:", EndTime.month)
print("Current day:", EndTime.day)

from datetime import datetime
DateEnd= datetime.now().time()
print('DateEnd : ', DateEnd)
print("hour =", DateEnd.hour)
print("minute =", DateEnd.minute)
print("second =", DateEnd.second)
print("microsecond =", DateEnd.microsecond)

'''
# Datetime end
DateEnd = datetime.now()
DateEnd.strftime("%y-%m-%d")
print("End camera :", DateEnd.strftime("%y-%m-%d"))
# Datetime end
EndTime = datetime.now()
EndTime.strftime("%H:%M:%S")
print("End camera :", EndTime.strftime("%H:%M:%S"))
'''

# Enlapsed time (s)
end_time = time.time()
Enlapsedtime = (end_time - start_time)
print("Temps actif: %0.2f" % Enlapsedtime, "secondes")
# Enlapsed time format hh:mm:ss
deltatime = str(timedelta(seconds=Enlapsedtime))
print("Temps actif: ", deltatime,'\n')

'''
# creation file csv Distance
with open('DistanceData3.csv','w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'Distance_AVG', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'DateStart' : BegTime,'Start_camera' : BegDate, 'Distance_AVG' : "{0:.2f}".format(Distance_AVG), 
    'DateEnd' : DateEnd,'End_camera' : EndTime, 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})

# Write line csv
with open('DistanceData3.csv','a', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'Distance_AVG', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'DateStart' : BegTime,'Start_camera' : BegDate, 'Distance_AVG' : "{0:.2f}".format(Distance_AVG), 
    'DateEnd' : DateEnd,'End_camera' : EndTime, 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})
'''
'''
# creation file csv Pitch, Roll, Yaw
with open('NewFile.csv','w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'pitch', 'yaw', 'roll', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'DateStart' : BegTime,'Start_camera' : BegDate, 'pitch' : pitch, 'yaw' : yaw, 'roll' : roll,
    'DateEnd' : DateEnd,'End_camera' : EndTime, 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})

# Write line csv
with open('PoseData.csv','a', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['DateStart', 'Start_camera', 'pitch', 'yaw', 'roll', 'DateEnd','End_camera', 'Temps_actif', 'Temps_actif_format']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'DateStart' : BegTime,'Start_camera' : BegDate, 'pitch' : pitch, 'yaw' : yaw, 'roll' : roll,
    'DateEnd' : DateEnd,'End_camera' : EndTime, 'Temps_actif' : "{0:.2f}".format(Enlapsedtime), 'Temps_actif_format' : deltatime})
'''

'''
# creation file csv Distance
with open('DataWebtrack2.csv','w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['BegTime', 'BegDate', 'Distance_AVG', 'ListTime', 'ListPitch','ListYaw', 'ListRoll', 'EndTime', 'DateEnd']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'BegTime' : BegTime,'BegDate' : BegDate, 'Distance_AVG' : Distance_AVG, 'ListTime' : timenow,
    'ListPitch' : pitch2,'ListYaw' : yaw, 'ListRoll' : roll, 'EndTime' : EndTime, 'DateEnd' : DateEnd})
'''

# Write line csv
with open('DataWebtrack2.csv','a', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['BegTime', 'BegDate', 'Distance_AVG', 'ListTime', 'ListPitch','ListYaw', 'ListRoll', 'EndTime', 'DateEnd']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'BegTime' : BegTime,'BegDate' : BegDate, 'Distance_AVG' : Distance_AVG, 'ListTime' : timenow,
    'ListPitch' : pitch2,'ListYaw' : yaw, 'ListRoll' : roll, 'EndTime' : EndTime, 'DateEnd' : DateEnd})


'''
# creation file csv Distance
with open('DataWebtrack.csv','w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['BegTime', 'BegDate', 'ListDist', 'listeTime', 'ListPitch','ListYaw', 'ListRoll', 'EndTime', 'DateEnd']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'BegTime' : BegTime,'BegDate' : BegDate, 'ListDist' : listeDist, 'listeTime' : listeTime,
    'ListPitch' : listePitch,'ListYaw' : listeYaw, 'ListRoll' : listeRoll, 'EndTime' : EndTime, 'DateEnd' : DateEnd})
'''

# Write line csv
with open('DataWebtrack.csv','a', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['BegTime', 'BegDate',  'ListDist', 'listeTime', 'ListPitch','ListYaw', 'ListRoll', 'EndTime', 'DateEnd']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'BegTime' : BegTime,'BegDate' : BegDate, 'ListDist' : listeDist, 'listeTime' : listeTime, 
    'ListPitch' : listePitch,'ListYaw' : listeYaw, 'ListRoll' : listeRoll, 'EndTime' : EndTime, 'DateEnd' : DateEnd})

print('image_points: \n', image_points)

# Close device
capVid.release()
# out.release()
cv.destroyAllWindows()
