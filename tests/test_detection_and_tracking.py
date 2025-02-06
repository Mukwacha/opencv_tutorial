from detection_and_tracking import DetectionAndTracking, VideoCaptureException
import os
import unittest

class TestDetectionAndTracking(unittest.TestCase):
    def setUp(self):
        """ Initializing model paths"""
        self.model_prototxt = '/data/models/MobileNetSSD_deploy.prototxt.txt'
        self.model_path = '/data/models/MobileNetSSD_deploy.caffemodel'

    def test_cars(self):
        """test processing for cars.mp4 """

        args = {'video': '/data/cars.mp4', 
                     'confidence': 0.2, 
                     'output': True}
        output_file = '/data/result_cars.avi'

        if os.path.isfile(output_file):
            os.remove(output_file)

        my_detector_and_tracker = DetectionAndTracking(self.model_prototxt, self.model_path, args=args)
        my_detector_and_tracker.process_video()

        self.assertTrue(os.path.isfile(output_file))

    def test_surveillance(self):
        """test processing for surveillance.mp4 """
        # TODO: write test using surveillance.mp4
        args ={'video':'/data/surveillance.mp4',
                'confidence':0.2,
                'output':True}
        output_file='/data/result_surveillance.avi'

        if os.path.isfile(output_file):
            os.remove(output_file)

        my_detector_and_tracker= DetectionAndTracking(self.model_prototxt,self.model_path,args=args)
        my_detector_and_tracker.process_video()

        self.assertTrue(os.path.isfile(output_file))

    def test_no_path(self):
        '''test handing of video path that do not exist'''
        args = {'video': '/data/no_path/cars.mp4', 
                     'confidence': 0.2, 
                     'output': True}

        my_detector_and_tracker = DetectionAndTracking(self.model_prototxt, self.model_path, args=args)

        with self.assertRaises(VideoCaptureException):
            my_detector_and_tracker.process_video()

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
