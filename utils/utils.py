import math
import cv2

def euclidian_dist(p1,p2):
    x0=p1[0]
    y0=p1[1]
    x1=p2[0]
    y1=p2[1]
    return math.sqrt((x1-x0)**2+(y1-y0)**2) 
#========================================================
def get_bbox(bb,id=False):
    x1=int(bb[0])
    y1=int(bb[1])
    x2=int(bb[2])
    y2=int(bb[3])
    width=x2-x1
    height=y2-y1

    if(id): 
        id_trk = int(bb[4])
        return  x1,y1,width,height,id_trk
    else : return x1,y1,width,height
#========================================================
def centroid2rect(x,y,w,h):
    left =int( x - w/2)
    top=int( y - h/2)
    right =int( left + w)
    bottom =int( top + h)
    start_point=(left,top)
    end_point=(right,bottom)
    return [start_point,end_point]
#========================================================
def bbox_inception(bbx1,bbx2,tol=30):

    boundb_x = bbx1[0]
    boundb_y = bbx1[1]
    boundb_w = bbx1[2]
    boundb_h = bbx1[3]

    tol_boundb_x = bbx1[0] + tol
    tol_boundb_y = bbx1[1] + tol
    tol_boundb_w = bbx1[2] - 2*tol
    tol_boundb_h = bbx1[3] - 2*tol

    innerb_x = bbx2[0]
    innerb_y = bbx2[1]
    innerb_w = bbx2[2]
    innerb_h = bbx2[3]

    flag1=False # first flag check if bbox is inside upper bound
    flag2=False # second flag check if bbox is outside under bound

    # If top-left inner box corner is inside the bounding box
    if boundb_x < innerb_x and boundb_y < innerb_y:
        # If bottom-right inner box corner is inside the bounding box
        if innerb_x + innerb_w < boundb_x + boundb_w \
                and innerb_y + innerb_h < boundb_y + boundb_h:
            flag1=True
        else: flag1 = False

    # If top-left inner box corner is inside the bounding box
    if tol_boundb_x > innerb_x and tol_boundb_y > innerb_y:
        # If bottom-right inner box corner is inside the bounding box
        if innerb_x + innerb_w > tol_boundb_x + tol_boundb_w \
                and innerb_y + innerb_h > tol_boundb_y + tol_boundb_h:
            flag2=True
        else: flag2 = False

    if(flag1 and flag2) : return True
    else : return False
#========================================================  
def bbox_IOU(retina_box, boxB,iou_th=0.8):
    '''
        Retina bounding box is [left, top, width, right]
        OpenCV bounding box is [left, top, width, right]
    '''
    #https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    xA = max(retina_box[0], boxB[0])
    yA = max(retina_box[1], boxB[1])
    xB = min(retina_box[0] + retina_box[2], boxB[0] + boxB[2])
    yB = min(retina_box[1] + retina_box[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = retina_box[2] * retina_box[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    #print(iou)
    if(iou> iou_th): return True
    else: return False
#========================================================
def draw_bboxes(bbs,frame, areas=False,trk=False):

    if(areas): Areas=[]
    for i in range(len(bbs)):
        bb=bbs[i]
        if(trk==False):x1,y1,width,height = get_bbox(bb)
        else:  x1,y1,width,height,Trk_id = get_bbox(bb,id=True)
        #print(f" x1: {x1} , x2: {x2}, y1: {y1} y2: {y2}")
        right=x1+width
        bottom=y1+height
        start=(x1,y1)
        end=(right,bottom)
        if (areas): Areas.append(width*height)
        frame=cv2.rectangle(frame,start,end,(0,0,255),2)
        if(trk==True): frame=cv2.putText(frame,f"ID: {Trk_id}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    if (areas): return frame,Areas 
    else : return frame
#========================================================
def bbox_area_filter(areas,bbs,frame,trk=False):
    #get the biggest bounding box
    index=areas.index(max(areas))
    bb=bbs[index]
    if(trk==False): x1,y1,width,height = get_bbox(bb)
    else : x1,y1,width,height,trk_id = get_bbox(bb,id=True)
    right=x1+width
    bottom=y1+height
    start=(x1,y1)
    end=(right,bottom)
    frame=cv2.rectangle(frame,start,end,(0,255,0),2)

    if(trk): return [x1,y1,width,height,trk_id],frame
    else : return [x1,y1,width,height],frame
#=======================================================
def show_in_frame(frame,text,counter,sys,color):
    if(color=="green"): c = (0,255,0)
    elif(color == "red"): c = (0,0,255)
    elif(color == "yellow"): c = (45,255,255)
    frame =cv2.putText(frame,text,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,c,2,cv2.LINE_AA)
    counter+=1
    if(counter>100):
        sys=0
        counter=0
    return sys,counter,frame