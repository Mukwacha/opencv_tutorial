# import the necessary packages
import argparse
import cv2
from loguru import logger
import numpy as np


# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
# NOTE: read first frame from video cars.mp4. hint: create cv2 video capture object and read first frame if ret is true else do nothing or exit the program using sys library (sys.exit(0)): ret, frame = cap.read()
 # VideoCapture object for vide


# NOTE: module to return video writer with args, cap as a parameter. hint: def get_video_writer(args, cap): return writer
# output video path


def _parse_args():
    """
    parse_args reads the command-line
    arguments and returns them as a namespace

    :params: None
    :return: argument namespace, argparse.Namespace
    """
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required= False, help="path to input image")
    ap.add_argument("-v", "--video", required=False,
        help="path to input video")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    ap.add_argument("-o", "--output", action='store_true',
        help="write annotated frames to output file")
    args = vars(ap.parse_args())

    return args


class VideoCaptureException(Exception):
    def __init__(self, message):
        super().__init__(message)
        # NOTE: pass message to exception superclass via initialization method of superclass. hint: single line call to superclass


class DetectionAndTracking:
    def __init__(self, model_prototxt, model, args=_parse_args()):
        self.video_name = args['video']
        self.confidence = args['confidence']
        self.output = args['output']
        logger.info("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(model_prototxt, model)

    def __get_video_writer(self, w, h, fps):
        """
        Creates and returns video writer object with args

        :pram args: Command-line arguments 
        :pram cap: OpenCV video capture object
        :returns: Videowriter object
        """
        
        output_path, output_filename = self.video_name.replace('.mp4','.avi').rsplit('/', 1)
        output_path = f"{output_path}/result_{output_filename}"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

        # NOTE: use cap.set to set the capture pointer to every fifth frame. hint: increment N by 5
        # read each frame
        return writer


    def __get_detections(self, frame, w, h):
        """
        get_detections takes a frame and passes
        it to a model and makes predictions

        :param frame: 3-channel image, numpy.ndarray
        :param w: width od frame
        :param h: height of frame
        :returns: predictions, python list
        """   
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,(300, 300), 127.5)
        logger.info("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        predictions = []

        for i in np.arange(0, detections.shape[2]):
             # extract the confidence (i.e., probability) associa    ted with the
             # prediction
             confidence = detections[0, 0, i, 2]
             # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
             if confidence > self.confidence:
                 # extract the index of the class label from the `detections`,
                 # then compute the (x, y)-coordinates of the bounding box for
                 # the object
                 idx = int(detections[0, 0, i, 1])
                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                 (startX, startY, endX, endY) = box.astype("int")
                 # add the prediction

                 predictions.append([startX, startY, endX, endY, confidence, idx])
        
        return predictions


    def __annotate(self, frame, startX, startY, endX, endY, COLORS, CLASSES, idx, confidence):
        """
        annotate annotates the frame with the bounding
        box information
        :param: frame
        :param: startX, int
        :param: startY, int
        :param: endX, int
        :param: endY, int
        :param: COLORS, numpy.ndarray
        :returns frame: frame with annotations, numpy.ndarray
        """
        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        logger.info(f"[INFO] {label}")
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
        
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        return frame


    # tracker = cv2.TrackerCSRT_create()
    # tracker.init(frame, initBB)
    # success, box = tracker.update(frame)
    def process_video(self):
        """
        Reads frames from a video, runs object detection and tracking, and writes output.
        :param net: Preloaded deep learning model
        :param args: Command-line arguments
        param COLORS: List of colors for each class
        """ 
        cap = cv2.VideoCapture(self.video_name)
        #cheking if video successufully opened
        if not cap.isOpened():
            raise VideoCaptureException("Error : video failed to open")

        # NOTE: supply videowriter with required parameters. hint: get the parameters from the capture object
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # NOTE: consider how N will affect fps of output video
        fps = int((cap.get(cv2.CAP_PROP_FPS)))
        
        if self.output:
            # NOTE: add function calls here ...
            writer = self.__get_video_writer(w, h, fps)

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        N = 5
        frame_count=0
        
        while True:
            # Setting capture pointer to frame_count position
            ret, frame = cap.read()
            if not ret:
                break
            
            logger.info(f'[INFO] Frame: {frame_count}')
            if (frame_count % N) == 0:
                # prediction stage every N frames
                # the important things in this stage are:
                # - get detections from the prediction model
                # - initialize tracker with start point for tracking
                logger.info('[INFO] Tracker initalisation ...')
                detections = self.__get_detections(frame, w, h)
            
                # tracker_ls.clear()
                tracker_ls =[]
            
                # loop over the detections
                for detection in detections:
                    startX, startY, endX, endY, confidence, idx = detection
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (startX, startY, abs(startX-endX), abs(startY-endY)))
                    
                    tracker_ls.append(tracker)
                    
                    if self.output:
                        frame = self.__annotate(frame, startX, startY, endX, endY, COLORS, CLASSES, idx, confidence)
               
            else:
                # tracking stage every N-1 out of N frames
                # the important things in this stage are:
                # - update tracker position
                # NOTE: get future value of prediction bbox by passing frame to tracker
                # NOTE: only change bounding box startX, startY, endX, endY and not confidence and idx
                logger.info('[INFO] Tracking ...')
                for i in range(len(tracker_ls)):
                    # NOTE: use tracker_ls[i] to update tracker, and after that use the new tracker coordinates
                    success, box= tracker_ls[i].update(frame)#updating tracker
                    if success:
                        #extracting new bounding box coordinates
                        (startX, startY, width,height)=[int(v) for v in box]
                        endX=startX +width
                        endY=startY +height
                        # NOTE: use detections[i] to make reference to confidence and idx
                        confidence= detections[i][-2]
                        idx= detections[i][-1]
            
                        if self.output:
                            frame = self.__annotate(frame, startX, startY, endX, endY, COLORS, CLASSES, idx, confidence)
            
                        # NOTE: write output frame to writer
            if self.output:
                writer.write(frame)
            #incrementing frame counter to the next Nth frame
            frame_count += 1

        cap.release()

        if self.output:
            writer.release()
            logger.info('[INFO] Output file written successfully.')

        logger.info('[INFO] Processing complete.')
