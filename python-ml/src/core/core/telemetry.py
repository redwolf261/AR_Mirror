import time
from collections import deque
import statistics

class PerformanceMonitor:
    """
    Real-time telemetry for monitoring pipeline stage performance.
    Tracks FPS, average, min, and max processing times.
    """
    def __init__(self, window_size=30):
        self.metrics = {}
        self.window_size = window_size
        self.start_times = {}

    def start_timer(self, stage_name: str):
        """Start timing a specific stage"""
        self.start_times[stage_name] = time.perf_counter()

    def stop_timer(self, stage_name: str):
        """Stop timing and record the duration"""
        if stage_name in self.start_times:
            elapsed = (time.perf_counter() - self.start_times[stage_name]) * 1000.0 # ms
            self.record_metric(stage_name, elapsed)
            del self.start_times[stage_name]

    def record_metric(self, name: str, value: float):
        """Record a raw metric value (e.g., ms duration)"""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(value)

    def get_stats(self, name: str):
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {'avg': 0, 'min': 0, 'max': 0}
        
        values = list(self.metrics[name])
        return {
            'avg': statistics.mean(values),
            'min': min(values),
            'max': max(values)
        }
    
    def get_fps(self, name: str):
        """Estimate FPS based on average frame time of a stage"""
        stats = self.get_stats(name)
        if stats['avg'] > 0:
            return 1000.0 / stats['avg']
        return 0.0

    def print_report(self):
        """Print a summary report of all metrics"""
        print("\n--- Telemetry Report ---")
        for name in self.metrics:
            stats = self.get_stats(name)
            fps = self.get_fps(name)
            print(f"{name:<20}: {stats['avg']:>6.2f}ms | {fps:>5.1f} FPS | (Min: {stats['min']:.1f}, Max: {stats['max']:.1f})")
        print("------------------------\n")
