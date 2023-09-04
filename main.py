#Title: zdpt ORB anti-spoofing
#Author: Fischer @ Cyberlabs
#Date: 22/07/2021
#Usage: python3 main_orb.py --source [SOURCE]

import numpy as np
import argparse
import sys
import cv2
import time
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Dolly.dolly_class import Dolly_zoom
from retinaface.retinaface import RetinaFace
import csv
from Sort.sort import *
from utils.utils import * 

def calc_hist(image_gray, mask=None):
    ''' Calculates histogram and dark and light qualities. '''
    hist = cv2.calcHist([image_gray], [0], mask, [256], [0, 256])

    dark = hist[0:64].sum() / hist.sum()
    light = hist[192:256].sum() / hist.sum()

    return dark, light
#========================================================
#==========================MAIN==========================
#========================================================
def main():
    print("Starting Zoom double picture")

    #parsing arguments--------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--vid",type=str,help="path to input video")
    parser.add_argument("--source",type=int, help="camera source")
    args=parser.parse_args()
    
    #check parsing arguments--------
    if(args.vid is not None):
        print(f"Source input: {args.vid}")  
        source =cv2.VideoCapture(args.vid) #get image from webcam
    else:
        if (args.source is None): 
            source =cv2.VideoCapture(0) #get image from webcam
            print("Source input: webcam")
        else:
            source =cv2.VideoCapture(args.source) #get image from webcam
            print(f"Source input: {args.source}")
    #-------------------------------------
    #create dolly zoom object
    #dolly = Dolly_zoom(control=True)

    #anti-spoofing threshold
    th= 0.4
    keypoint_th = 300 # number of keypoint in second picture
    dark_th = 0.6

    #create instance of SORT
    mot_tracker = Sort() 

    #retina face detector
    retina=RetinaFace('retinaface/mnet.25', 0, 0, 'net3')

    #setpoint bbox
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH)) # width of whole image
    height= int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height of whole image
    x_mid=width/2 # image middle point X
    y_mid=height/2 # image middle point Y
    w1=170 #width of first setpoint
    h1=230 #height of first setpoint
    w2=250 # width of second setpoint
    h2=340 # height of second setpoint
    s1,e1=centroid2rect(x_mid,y_mid,w1,h1) # first bbox 
    s2,e2=centroid2rect(x_mid,y_mid,w2,h2) # second bbox
    setpoint_color = (255,0,0) # setpoint bbox color
    set_bbox1=[x_mid-w1/2,y_mid-h1/2,w1,h1] 
    set_bbox2=[x_mid-w2/2,y_mid-h2/2,w2,h2]

    frame_counter=0
    reset_frame_counter=0 # tracking reset id
    access_frame_counter=0 # counter for showing result

    #sys_flag
    # 0 - get first picture
    # 1 - get second picture
    # 2 - get metrics
    # 3 - show "Real face" on screen
    # 4 - show "Spoof" on screen
    # 5 - show "Bad Lighting" on screen
    sys_flag = 0
    
    while(True):
        start_time=time.time()
        ret,original_frame=source.read()

        #flip image on y axis
        original_frame = cv2.flip(original_frame,1)

        if(not ret): # frame is empty
            print("failed to get frame")
            break

        #detect face
        bbs,points= retina.detect(original_frame)

        #if no face is detected
        if(len(bbs)==0):
            cv2.imshow("frame",original_frame)      
            k=cv2.waitKey(30)
            if k == 27: break
            if(sys_flag==0):reset_frame_counter+=1
            if(reset_frame_counter>30): #after N frames without people 
                print("Tracking ids reset")
                reset_frame_counter=0 # reset counter
                mot_tracker.reset_count() # reset Sort ids
            continue

        #tracking
        # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
        track_bbs_ids = mot_tracker.update(bbs)
        
        frame=original_frame.copy()

        #draw the setpoint bboxes
        if(sys_flag==0):frame=cv2.rectangle(frame,s1,e1,setpoint_color,5)
        elif(sys_flag==1): frame=cv2.rectangle(frame,s2,e2,setpoint_color,5)
        elif(sys_flag==3): sys_flag,access_frame_counter,frame = show_in_frame(frame,"Real face", access_frame_counter, sys_flag,"green")
        elif(sys_flag==4): sys_flag,access_frame_counter,frame = show_in_frame(frame,"Spoof", access_frame_counter, sys_flag,"red")
        elif(sys_flag==5): sys_flag,access_frame_counter,frame = show_in_frame(frame,"Bad Lighting", access_frame_counter, sys_flag,"yellow")
        
        #draw bounding boxes
        frame,Areas = draw_bboxes(track_bbs_ids,frame,areas=True,trk=True)
        if(len(Areas)==0): continue

        #choose the bbox with the biggest area
        bbox_filtered,frame = bbox_area_filter(Areas,track_bbs_ids,frame,trk=True)

        #check if the face is positioned correctly for the picture 1
        if(bbox_IOU(bbox_filtered, set_bbox1) and sys_flag==0):
            #crop original image
            x1=int(set_bbox1[0])
            y1=int(set_bbox1[1])
            bottom=int(set_bbox1[1]+set_bbox1[3])
            right=int(set_bbox1[0]+set_bbox1[2])
            id1=int(bbox_filtered[4])
            cropped = original_frame[y1:bottom,x1:right]
            cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)

            print(f"id1: {id1}")

            #ORB feature extractor
            orb1=cv2.ORB_create(4000)
            equ_crop= cv2.equalizeHist(cropped) # https://docs.opencv.org/4.5.2/d5/daf/tutorial_py_histogram_equalization.html
            equ_crop = cv2.resize(equ_crop,(set_bbox2[2],set_bbox2[3]),interpolation = cv2.INTER_AREA )
            kp1,des1 = orb1.detectAndCompute(cropped,None)
            frame1=equ_crop.copy()
            frame1_light = cropped.copy()

            sys_flag=1

        #check if the face is positioned correctly for the picture 2
        elif(bbox_IOU(bbox_filtered, set_bbox2) and sys_flag==1):
            #crop original image
            x1=int(set_bbox2[0])
            y1=int(set_bbox2[1])
            bottom=int(set_bbox2[1]+set_bbox2[3])
            right=int(set_bbox2[0]+set_bbox2[2])
            id2 = int(bbox_filtered[4])
            cropped = original_frame[y1:bottom,x1:right]
            cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)


            print(f"id2: {id2}")

            #ORB feature extractor
            orb2=cv2.ORB_create(4000)
            equ_crop= cv2.equalizeHist(cropped) #https://docs.opencv.org/4.5.2/d5/daf/tutorial_py_histogram_equalization.html
            kp2,des2 = orb2.detectAndCompute(equ_crop,None)
            
            #frame2=cropped.copy()
            frame2 = equ_crop.copy()
            frame2_light = cropped.copy()

            sys_flag=2

        if(sys_flag==2): # calculate metrics
            #check if ids match
            if(id1==id2):
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)#,crossCheck=True)
                #matches = matcher.match(des1,des2)
                matches = matcher.knnMatch(des1,trainDescriptors=des2,k=2)
                good_matches = [m for (m, n) in matches if m.distance < 0.75 * n.distance]

                if(len(matches)==0): sys_flag = 5
                else:
                    similarity = len(good_matches)/len(matches)
                    
                    print(f"good matches:  {len(good_matches)}")
                    print(f"matches: {len(matches)}")
                    print(f"KeyPoints1: {len(des1)}  KeyPoints2: {len(des2)}")
                    print(f"similarity: {similarity}")

                    if(len(des2)>=keypoint_th): # check good matches first
                        if(similarity<th): # real face
                            print("real face")
                            sys_flag=3
                        else:
                            print("spoof")
                            sys_flag=4
                    else:
                        print("spoof")
                        dark1,light1 = calc_hist(frame1_light)
                        dark2,light2 = calc_hist(frame2_light)
                        print(f'frame 1 dark: {dark1} light: {light1}')
                        print(f'frame 2 dark: {dark2} light: {light2}')
                        if(dark1 >= dark_th or dark2>= dark_th): sys_flag = 5
                        else: sys_flag=4
                        
                    
                    print("===============================================")
                    match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches,None)
                    cv2.imshow("Match",match_img)
            else:
                print("ids dont match")
                print("===============================================")
                sys_flag=0
        
        
            
        frame_counter+=1

        cv2.imshow("frame",frame)      
        k=cv2.waitKey(30)
        if k == 27: break
        #print FPS
        #if(printf == ''): print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop 
        #else : print(printf + "  FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop 
        #print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop 

#================================================
if __name__=="__main__":
    main()