import threading
import queue
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
from .telemetry import PerformanceMonitor

class Stage(ABC):
    """
    Abstract Base Class for a pipeline stage.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

class AsyncPipeline:
    """
    Manages an asynchronous processing pipeline.
    Decouples input (producer) from processing (worker) from output (consumer).
    """
    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        self.input_queue = queue.Queue(maxsize=1) # Keep only latest frame
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.worker_thread = None
        self.monitor = monitor

    def start(self, processing_stage: Stage):
        """Start the worker thread with the given processing stage"""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, args=(processing_stage,), daemon=True)
        self.worker_thread.start()
        print(f"Pipeline started with stage: {processing_stage.name}")

    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            print("Pipeline stopped.")

    def submit(self, data: Any):
        """
        Submit data to the pipeline.
        Non-blocking: Overwrites old data if queue is full (Frame Dropping).
        """
        try:
            # Empty queue if full to ensure we process latest
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.input_queue.put_nowait(data)
        except queue.Full:
            pass # Should not happen due to clearing above

    def get_result(self) -> Optional[Any]:
        """
        Get the latest result.
        Non-blocking: Returns None if no new result is ready.
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self, stage: Stage):
        """Internal loop running in separate thread"""
        while self.running:
            try:
                # Wait for input with timeout to allow checking self.running
                input_data = self.input_queue.get(timeout=0.1)
                
                # Telemetry
                if self.monitor:
                    self.monitor.start_timer(stage.name)
                
                # Process
                result = stage.process(input_data)
                
                # Telemetry
                if self.monitor:
                    self.monitor.stop_timer(stage.name)
                
                # Push output (Overwrite old result)
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in pipeline worker {stage.name}: {e}")
                import traceback
                traceback.print_exc()

class ResourceManager:
    """
    Simple cache for heavy resources (images, models).
    """
    _cache = {}

    @classmethod
    def get(cls, key: str):
        return cls._cache.get(key)

    @classmethod
    def put(cls, key: str, value: Any):
        cls._cache[key] = value

    @classmethod
    def clear(cls):
        cls._cache.clear()
