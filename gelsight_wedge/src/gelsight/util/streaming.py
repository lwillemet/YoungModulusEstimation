import urllib.request
import urllib
import cv2
import numpy as np
import time 
from .processing import warp_perspective

class Streaming(object):
    def __init__(self, url, corners, output_sz, warp_enable):
        self.image = None
        self.warp_enable = warp_enable
        if warp_enable:
            self.warped_img = None
        self.url = url

        self.streaming = False
        self.corners = corners
        self.output_sz = output_sz

        self.start_stream()

    def __del__(self):
        self.stop_stream()

    def start_stream(self):
        self.streaming = True
        self.stream=urllib.request.urlopen(self.url)

    def stop_stream(self):
        if self.streaming == True:
            self.stream.close()
        self.streaming = False

    def load_stream(self):
        stream = self.stream
        bytess=b''

        while True:
            if self.streaming == False:
                time.sleep(0.01)
                continue

            bytess+=stream.read(32767)

            a = bytess.find(b'\xff\xd8') # JPEG start
            b = bytess.find(b'\xff\xd9') # JPEG end

            if a!=-1 and b!=-1:
                jpg = bytess[a:b+2] # actual image
                bytess= bytess[b+2:] # other informations

                self.image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
                if self.warp_enable:
                    self.warped_img = warp_perspective(self.image, self.corners, self.output_sz)
