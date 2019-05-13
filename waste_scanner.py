import os
from imageai.Detection import VideoObjectDetection
import cv2
import serial, time
from matplotlib import pyplot as plt
import numpy as np

global recyclingCount
global compostCount
global landfillCount
global ct
global itemCount
global prevItemCount
global noItemsInFrame
global objectsTotalClassified
global connected
global prevTime

execution_path = os.getcwd()

arduinoSerial = '/dev/cu.usbmodem143101'
virtualSerial = '/dev/ttyp3' #virtual serial

connected = os.system('ls ' + arduinoSerial)

if connected == 0:
    print("Arduino connected at " + arduinoSerial)
    arduino = serial.Serial(arduinoSerial, 9600)  # open serial port
else:
    print("USING Virtual Serial at " + virtualSerial)
    arduino = serial.Serial(virtualSerial, 9600)  # open serial port

time.sleep(2)

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "models/yolo.h5"))
detector.loadModel(detection_speed="fast")


# 1 = recycling, 2 = compost, 3 = landfill
dict = {
    "bottle": b'1', "cup": b'3', "apple": b'2',"spoon": b'2',"fork": b'2',
    "knife": b'2',"banana": b'3',"pizza": b'2',"orange": b'2',
    "bowl": b'3', "cell phone": b'1', "broccoli": b'3', "wine glass": b'3',
    "donut": b'3', "cake": b'3', "scissors": b'3', "person": b'2'
}

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
               'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold',
               'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry',
               'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt',
               'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock',
               'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey',
               'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen',
               'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki',
               'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood',
               'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
               'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver',
               'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink',
               'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple',
               'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue',
               'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock',
               'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream',
               'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew',
               'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow',
               'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

resized = False

noItemsInFrame = np.ones((5),dtype=bool)

prevItemCount = 0
recyclingCount = 0
compostCount = 0
# keep this as a small number that way we don't get division by 0 errors
landfillCount = 0.00001

def barPlot(recyclingCount,compostCount,landfillCount):
    plt.subplot(1, 2, 2)
    wasteCount = (round(recyclingCount), round(compostCount), round(landfillCount))
    bars = ('Recycling', 'Compost', 'Landfill')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, wasteCount, color=['blue', 'green', 'gray'])
    plt.xticks(y_pos, bars)
    divrate = (recyclingCount + compostCount) / (landfillCount + recyclingCount + compostCount)
    plt.title("Diversion Rate: " + str(round(divrate, 1) * 100) + "%")
    plt.pause(0.01)

def frame_func(frame_number, output_array, output_count,returned_frame):
    global recyclingCount
    global compostCount
    global landfillCount
    # global ct
    global itemCount
    global prevItemCount
    global noItemsInFrame
    global objectsTotalClassified
    global currTime
    global prevTime

    # initialize number of frames of where we allow no items
    plt.clf()
    this_colors = []
    labels = []
    sizes = []
    counter = 0

    print("frame_num = \n", frame_number,'\n', "output_array = \n", output_array, '\n', "output_count = \n", output_count)

    # ct = CentroidTracker()

    rects = []

    for item in range(0,len(output_array)):
        arrayitem =  output_array[item]
        box = arrayitem["box_points"]
        box = np.asarray(box)
        rects.append(box.astype("int"))

    itemCount = len(output_array)
    
    for eachObject in output_count:
        objectName = eachObject
        print(objectName)
        bin = dict.get(objectName, 0)
        if itemCount > prevItemCount:
            if (bin == b'1'):
                recyclingCount = recyclingCount + 1
                # arduino.write(bin)
            elif (bin == b'2'):
                compostCount = compostCount + 1
                # arduino.write(bin)
            elif (bin == b'3'):
                landfillCount = landfillCount + 1
                # arduino.write(bin)
        print("object:" + objectName)
        print("bin = " + "bin")
        print(bin)

        counter += 1
        labels.append(eachObject + " = " + str(output_count[eachObject]))
        sizes.append(output_count[eachObject])
        this_colors.append(color_index[eachObject])

        # arduino.write(bin)

    if itemCount == 0 and np.all(noItemsInFrame==True):
        noItemsInFrame = np.append(noItemsInFrame,True)
        noItemsInFrame = noItemsInFrame[1:]
        prevItemCount = 0
        # print("case 1")
    elif itemCount == 0:
        noItemsInFrame = np.append(noItemsInFrame,True)
        noItemsInFrame = noItemsInFrame[1:]
        # print("case 2")
    else:
        prevItemCount = itemCount
        noItemsInFrame = np.append(noItemsInFrame,False)
        noItemsInFrame = noItemsInFrame[1:]
        # print("case 3")
    # print(noItemsInFrame)

    global resized
    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(1000, 500)
        resized = True
    plt.subplot(2, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    newplotframe = cv2.cvtColor(returned_frame, cv2.COLOR_RGB2BGR)
    plt.imshow(newplotframe, interpolation="none")

    plt.subplot(2, 2, 3)
    plt.title("Identified Objects")
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    # bar chart analytics:
    barPlot(recyclingCount, compostCount, landfillCount)


    # take current time
    currTime = time.time()

    # Time elapsed
    seconds = currTime - prevTime
    print("Time taken : {0} seconds".format(seconds))
    print("______________")
    prevTime = currTime


prevTime = time.time()

#detect object function
custom_objects = detector.CustomObjects(cup=True, apple=True, banana=True, fork=True, spoon=True, bottle=True, knife=True,pizza=True,orange=True,bowl=True, cell_phone=True,broccoli=True,wine_glass=True,donot=True,cake=False,scissors=True, person = True)

detections = detector.detectCustomObjectsFromVideo(custom_objects = custom_objects,camera_input=camera,per_frame_function=frame_func,
                                 output_file_path=os.path.join(execution_path, "video_frame_analysis")
                                 , frames_per_second=30, log_progress=True, minimum_percentage_probability=55,return_detected_frame=True)

# detections = detector.detectObjectsFromVideo(camera_input=camera,
#                                       output_file_path=os.path.join(execution_path, "video_frame_analysis"),
#                                       frames_per_second=30, per_frame_function=frame_func,
#                                       minimum_percentage_probability=60, return_detected_frame=True)

# detections = detector.detectObjectsFromVideo(camera_input=camera,
#                                 output_file_path=os.path.join(execution_path, "camera_detected_video")
#                                 , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)

# print(vars(detections))

