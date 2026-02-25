import cv2
import threading
import time

class AsyncCamera:
    """
    Threaded camera capture to always have the latest frame ready.
    Reduces latency by decoupling capture rate from processing rate.
    """
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        
        # FPS calculation
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Daemon thread exits when main program exits
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # FPS tracking
            self.frame_count += 1
            if time.time() - self.start_time > 1:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.frame_count = 0
                self.start_time = time.time()
                
            time.sleep(0.005) # Prevent CPU hogging (approx 200fps cap check)

    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return None
            return self.frame.copy()

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
