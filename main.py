import torch
import time
import cv2
import numpy as np
import pafy
#pafy used to read video from youtube

from torch import hub
#Hub contains various models

model = torch.hub.load( \
                      'ultralytics/yolov5', \
                      'yolov5s', \
                      pretrained=True)

cap = cv2.VideoCapture('Yolo_Stream_Videos/Mass Effect_Legendary_Edition_Trailer.mp4')




def runDetect(self):
    #videoPlayer = self.get_video_stream()
    videoPlayer = cap
    assert videoPlayer.isOpened()

    # Below creates video writer object to write output strea,

    x_shape = int(videoPlayer.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(videoPlayer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")
    # Using MJPG codex

    outputFilename = "Test_File"
    output = cv2.VideoWriter(outputFilename, four_cc, 20, (x_shape, y_shape))

    ret, frame = videoPlayer.read()
    while ret:
        startTime = time.time()
        results = self.scoreFrame(frame, model)
        frame = self.plot_boxes(results, frame)
        endTime = time.time()

        fps = 1 / np.round(endTime - startTime, 3)
        print(f"Frames Per Second: {fps}")
        output.write(frame)
        ret, frame = videoPlayer.read()

def scoreFrame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [torch.tensor(frame)]
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cordinates = results.xyxyn[0][:, :-1].numpy()
    return labels, cordinates
#Gets results from frame and sends back labels and coordinates

def plot_boxes(self, results, frame):
    labels, cordinates = results
    n = len(labels)
    x_shape, y_shape =  frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cordinates[i]

        if row[4] < 0.2:
            continue
        #Will not make prediction if confidence is lower than 0.2

        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)

        BOX_COLOUR = (0, 255,0)
        classes = self.model.names
        LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (x1,y1), (x2,y2), (BOX_COLOUR,2))
        cv2.putText(frame, classes[labels[i]], (x1, y1), LABEL_FONT, 0.9, BOX_COLOUR, 2)
        #Augment image with info detected by model

        return frame

newFrame = cv2.imread('Test_Photos/wallpaper.jpg')
print(scoreFrame(newFrame, model))


'''
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
