from ultralytics import YOLO  
import cv2
import math
model = YOLO("yolov8s.pt")  



def boxCenter(coords):
    [left, top, right, bottom] = coords
    return [(left+right)/2,(top+bottom)/2]

def closestBox(boxes, coords):
    distance = []
    center = boxCenter(coords)
    for box in boxes:
        boxCent = boxCenter(box.xyxy[0].numpy().astype(int))
        distance.append(math.dist(boxCent,center))
    return boxes[distance.index(min(distance))]


def adjustBoxSize(coords, box_width, box_height):
    [centerX, centerY] = boxCenter(coords)
    return [centerX-box_width/2, centerY-box_height/2, centerX+box_width/2, centerY+box_height/2]



def adjustBoundaries(coords, screen):
    [left, top, right, bottom] = coords
    [width, height]=screen
    if left<0:
        right=right-left
        left=0
    if top<0:
        bottom=bottom-top
        top=0
    if right>width:
        left=left-(right-width)
        right=width
    if bottom>height:
        top=top-(bottom-height)
        bottom=height
    return [round(left), round(top), round(right), round(bottom)]



fileSource = 'highlights (online-video-cutter.com).mp4' 
fileTarget = 'test_vide0_processed.mp4' 
cropCoords = [100,100,500,500] 


vidCapture = cv2.VideoCapture(fileSource)
fps = vidCapture.get(cv2.CAP_PROP_FPS)
totalFrames = vidCapture.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not cropCoords:
    [box_left, box_top, box_right, box_bottom] = [0, 0, width, height]
else:
    [box_left, box_top, box_right, box_bottom] = cropCoords
    if (box_left<0):
        box_left=0
    if (box_top<0):
        box_top=0
    if (box_right)>width:
        box_right=width
    if (box_bottom>height):
        box_bottom=height
lastCoords = [box_left, box_top, box_right, box_bottom]
lastBoxCoords = lastCoords
box_width = box_right-box_left
box_height = box_bottom-box_top


outputWriter = cv2.VideoWriter(fileTarget, cv2.VideoWriter_fourcc(*'MPEG'), fps, (box_width, box_height))


frameCounter = 1
while True:

    r, im = vidCapture.read()

    if not r:
        print("Video Finished!")
        break

    print("Frame: "+str(frameCounter))
    frameCounter = frameCounter+1
    results = model.predict(source=im, conf=0.3, iou=0.2, device='0') 
    #print(results)
    boxes = results[0].boxes 
    box = closestBox(boxes, lastBoxCoords) 
    if box is not None:
        lastBoxCoords = box.xyxy[0].numpy().astype(int) 

        newCoords = adjustBoxSize(box.xyxy[0].numpy().astype(int), box_width, box_height)
        newCoords = adjustBoundaries(newCoords,[width, height])     
        [box_left, box_top, box_right, box_bottom] = newCoords
        imCropped = im[box_top:box_bottom, box_left:box_right]

        outputWriter.write(imCropped) 

    
vidCapture.release()
outputWriter.release()