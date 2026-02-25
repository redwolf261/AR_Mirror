"""
RTX 2050 Performance Benchmarks for 3D Virtual Try-On

Measures performance of:
1. SMPL body reconstruction
2. Mesh garment wrapping
3. Physics simulation
4. Rendering pipeline
5. End-to-end latency

Target: 30 FPS real-time, 5 FPS with diffusion refinement

Author: Rivan Avinash Shetty (@redwolf261)
Date: February 14, 2026
Hardware: RTX 2050 4GB, i7-12650H
"""

import numpy as np
import torch
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple
import psutil  # type: ignore[import-untyped]

from src.core.smpl_body_reconstruction import SMPLBodyReconstructor, SMPLMeshResult
from src.core.mesh_garment_wrapper import GarmentMesh, MeshGarmentWrapper, PhysicsSimulator


def _make_synthetic_body_mesh(num_vertices: int = 6890) -> SMPLMeshResult:
    """Create a synthetic body mesh for benchmarking (no SMPL model required)."""
    np.random.seed(42)
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    for i in range(num_vertices):
        t = i / num_vertices
        y = t * 1.7
        width = 0.2 if 0.3 < t < 0.6 else 0.08
        vertices[i] = [width * np.sin(i * 0.1), y, width * np.cos(i * 0.1)]

    faces = []
    for i in range(0, num_vertices - 2, 3):
        faces.append([i, i + 1, i + 2])
    faces = np.array(faces, dtype=np.int32)

    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        normals[face] += n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    return SMPLMeshResult(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uv_coords=None,
        shape_params=np.zeros(10, dtype=np.float32),
        pose_params=np.zeros(72, dtype=np.float32),
    )


class PerformanceBenchmark:
    """Performance benchmark suite for RTX 2050."""
    
    def __init__(self, device: str = 'cuda', num_warmup: int = 5, num_iterations: int = 100):
        self.device = device
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        
        # Initialize components
        self.wrapper = MeshGarmentWrapper(device=device)
        self.physics = PhysicsSimulator(device=device)
        
        # Synthetic body mesh (always available)
        self.body_mesh = _make_synthetic_body_mesh()
        
        # Try real SMPL reconstructor
        self.smpl = SMPLBodyReconstructor(device=device)
        self.smpl_available = self.smpl.is_available
        
        # Generate test data
        self.landmarks = self._generate_synthetic_landmarks()
        self.garment_img, self.garment_mask = self._generate_synthetic_garment()
        
        self.results = {}
    
    def _generate_synthetic_landmarks(self) -> Dict:
        """Generate synthetic MediaPipe landmarks dict."""
        landmarks = {}
        positions = [
            (0.50, 0.20), (0.49, 0.18), (0.48, 0.18), (0.47, 0.18),
            (0.51, 0.18), (0.52, 0.18), (0.53, 0.18), (0.47, 0.20),
            (0.53, 0.20), (0.49, 0.22), (0.51, 0.22), (0.40, 0.40),
            (0.60, 0.40), (0.30, 0.55), (0.70, 0.55), (0.25, 0.70),
            (0.75, 0.70), (0.23, 0.72), (0.77, 0.72), (0.22, 0.71),
            (0.78, 0.71), (0.24, 0.73), (0.76, 0.73), (0.45, 0.65),
            (0.55, 0.65), (0.44, 0.80), (0.56, 0.80), (0.43, 0.95),
            (0.57, 0.95), (0.42, 0.97), (0.58, 0.97), (0.41, 0.98),
            (0.59, 0.98),
        ]
        for idx, (x, y) in enumerate(positions):
            landmarks[idx] = {'x': x, 'y': y, 'visibility': 0.95}
        return landmarks
    
    def _generate_synthetic_garment(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic garment image."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[50:200, 50:200] = [255, 0, 0]
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        return img, mask
    
    def _measure_time(self, func, *args, **kwargs) -> Dict[str, float]:
        """Measure execution time and memory usage."""
        times = []
        
        for _ in range(self.num_warmup):
            _ = func(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = func(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        times = np.array(times)
        
        result = {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'fps': float(1000.0 / times.mean())
        }
        
        if self.device == 'cuda':
            result['memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return result
    
    def benchmark_smpl_reconstruction(self):
        """Benchmark SMPL body reconstruction."""
        print("\n[1/8] Benchmarking SMPL Reconstruction...")
        
        if not self.smpl_available:
            print("  SMPL model not available — skipping (using synthetic mesh)")
            self.results['smpl_reconstruction'] = {
                'mean_ms': 0, 'std_ms': 0, 'min_ms': 0, 'max_ms': 0,
                'fps': float('inf'), 'note': 'skipped — model not present'
            }
            return
        
        def reconstruct():
            return self.smpl.reconstruct(
                landmarks=self.landmarks,
                frame_shape=(480, 640),
            )
        
        results = self._measure_time(reconstruct)
        self.results['smpl_reconstruction'] = results
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
        if 'memory_mb' in results:
            print(f"  Memory: {results['memory_mb']:.1f} MB")
    
    def benchmark_mesh_wrapping(self):
        """Benchmark garment mesh wrapping."""
        print("\n[2/8] Benchmarking Mesh Wrapping...")
        
        garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
        
        def wrap():
            return self.wrapper.wrap_garment(
                garment_mesh=garment_mesh,
                body_mesh=self.body_mesh,
                garment_type='tshirt',
            )
        
        results = self._measure_time(wrap)
        self.results['mesh_wrapping'] = results
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
        if 'memory_mb' in results:
            print(f"  Memory: {results['memory_mb']:.1f} MB")
    
    def benchmark_garment_mesh_generation(self):
        """Benchmark garment mesh generation from image."""
        print("\n[3/8] Benchmarking Garment Mesh Generation...")
        
        def generate():
            return GarmentMesh.from_image(self.garment_img, self.garment_mask)
        
        results = self._measure_time(generate)
        self.results['garment_generation'] = results
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
    
    def benchmark_rendering(self):
        """Benchmark software rendering."""
        print("\n[4/8] Benchmarking Rendering (Software Rasterization)...")
        
        garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
        wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
        
        camera_matrix = np.eye(3, dtype=np.float32)
        
        def render():
            return wrapped.render_to_image(
                camera_matrix=camera_matrix,
                image_size=(480, 640),
            )
        
        results = self._measure_time(render)
        self.results['rendering_software'] = results
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
    
    def benchmark_gpu_rendering(self):
        """Benchmark GPU hardware rendering (moderngl)."""
        print("\n[5/8] Benchmarking Rendering (GPU / moderngl)...")
        
        try:
            from src.core.gpu_renderer import create_renderer
            renderer = create_renderer(640, 480, shading='phong')
        except Exception as e:
            print(f"  GPU renderer not available: {e}")
            self.results['rendering_gpu'] = {'note': 'not available'}
            return
        
        garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
        wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
        
        def render_gpu():
            return renderer.render_wrapped_mesh(wrapped, image_size=(480, 640))
        
        results = self._measure_time(render_gpu)
        self.results['rendering_gpu'] = results
        renderer.release()
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
    
    def benchmark_physics(self):
        """Benchmark spring-force physics simulation step."""
        print("\n[6/8] Benchmarking Physics (Spring Forces + Collision)...")
        
        garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
        wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
        
        physics = PhysicsSimulator(device=self.device, grid_size=32)
        
        def simulate():
            return physics.simulate_step(wrapped, self.body_mesh, num_iterations=5)
        
        results = self._measure_time(simulate)
        self.results['physics'] = results
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
    
    def benchmark_temporal_cache(self):
        """Benchmark temporal cache hit/miss latency."""
        print("\n[7/8] Benchmarking Temporal Cache...")
        
        from src.core.temporal_cache import TemporalCache, CachedFrame
        cache = TemporalCache(motion_threshold=0.01, max_reuse_frames=5)
        
        # Build a landmark dict
        lm = {i: {'x': 0.5, 'y': float(i) / 33, 'visibility': 1.0} for i in range(33)}
        
        # First frame (miss) stores a result
        camera = np.eye(3, dtype=np.float32)
        garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
        wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
        rendered = wrapped.render_to_image(camera, (480, 640))
        cache.store(CachedFrame(
            timestamp=time.perf_counter(), landmarks=lm,
            body_mesh=self.body_mesh, wrapped_mesh=wrapped, rendered=rendered,
        ))
        
        # Measure cache HIT latency
        hit_times = []
        for _ in range(self.num_iterations):
            t0 = time.perf_counter()
            cache.should_recompute(lm)
            cache.get_cached()
            hit_times.append(time.perf_counter() - t0)
        
        hit_arr = np.array(hit_times) * 1000
        self.results['cache_hit'] = {
            'mean_ms': float(np.mean(hit_arr)),
            'std_ms': float(np.std(hit_arr)),
            'fps': float(1000.0 / np.mean(hit_arr)) if np.mean(hit_arr) > 0 else 0,
        }
        print(f"  Cache HIT: {np.mean(hit_arr):.4f}ms ({cache.hit_rate:.0%} hit rate)")
    
    def benchmark_end_to_end(self):
        """Benchmark complete end-to-end pipeline (with GPU renderer)."""
        print("\n[8/8] Benchmarking End-to-End Pipeline...")
        
        gpu_renderer = None
        try:
            from src.core.gpu_renderer import create_renderer
            gpu_renderer = create_renderer(640, 480, shading='flat')
        except Exception:
            pass
        
        camera_matrix = np.eye(3, dtype=np.float32)
        
        def pipeline():
            garment_mesh = GarmentMesh.from_image(self.garment_img, self.garment_mask)
            wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
            if gpu_renderer:
                output = gpu_renderer.render_wrapped_mesh(wrapped, image_size=(480, 640))
            else:
                output = wrapped.render_to_image(camera_matrix, (480, 640))
            return output
        
        results = self._measure_time(pipeline)
        self.results['end_to_end'] = results
        
        if gpu_renderer:
            gpu_renderer.release()
        
        print(f"  Mean: {results['mean_ms']:.2f}ms +/- {results['std_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")
        if 'memory_mb' in results:
            print(f"  Memory: {results['memory_mb']:.1f} MB")
    
    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("=" * 80)
        print("RTX 2050 Performance Benchmark Suite")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Warmup iterations: {self.num_warmup}")
        print(f"Benchmark iterations: {self.num_iterations}")
        
        if self.device == 'cuda' and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")  # type: ignore[attr-defined]
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        self.benchmark_smpl_reconstruction()
        self.benchmark_garment_mesh_generation()
        self.benchmark_mesh_wrapping()
        self.benchmark_rendering()
        self.benchmark_gpu_rendering()
        self.benchmark_physics()
        self.benchmark_temporal_cache()
        self.benchmark_end_to_end()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        print("\n| Component | Latency | FPS | Memory |")
        print("|-----------|---------|-----|--------|")
        
        for name, result in self.results.items():
            if result.get('note'):
                print(f"| {name.replace('_', ' ').title():20s} | {'N/A':>7s} | {'N/A':>5s} | {'N/A':>8s} |")
                continue
            mem_str = f"{result.get('memory_mb', 0):.0f} MB" if 'memory_mb' in result else "N/A"
            print(f"| {name.replace('_', ' ').title():20s} | {result['mean_ms']:6.1f}ms | {result['fps']:5.1f} | {mem_str:8s} |")
        
        print("\n" + "=" * 80)
        print("TARGET ANALYSIS")
        print("=" * 80)
        
        e2e = self.results['end_to_end']
        target_real_time = 30
        target_hq = 5
        
        print(f"\nReal-time Target: {target_real_time} FPS")
        print(f"Current: {e2e['fps']:.1f} FPS")
        
        if e2e['fps'] >= target_real_time:
            print("REAL-TIME TARGET MET")
        else:
            speedup_needed = target_real_time / e2e['fps']
            print(f"Need {speedup_needed:.1f}x speedup")
            print("Suggestions:")
            print("  - Enable TensorRT optimization")
            print("  - Use CUDA graphs")
            print("  - Reduce mesh resolution")
            print("  - Implement hardware rendering (OpenGL)")
        
        return self.results
    
    def save_results(self, output_path: str = "benchmarks/rtx2050_results.json"):
        """Save benchmark results to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_iterations': self.num_iterations,
            'num_warmup': self.num_warmup,
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            metadata['gpu'] = torch.cuda.get_device_name(0)
            metadata['cuda_version'] = torch.version.cuda  # type: ignore[attr-defined]
            metadata['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        data = {'metadata': metadata, 'results': self.results}
        
        with open(out, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to {out}")


class MemoryProfiler:
    """Profile memory usage during pipeline execution."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.wrapper = MeshGarmentWrapper(device=device)
        self.body_mesh = _make_synthetic_body_mesh()
    
    def profile_memory_timeline(self):
        """Profile memory usage at each pipeline stage."""
        print("\n" + "=" * 80)
        print("MEMORY PROFILING")
        print("=" * 80)
        
        if self.device != 'cuda' or not torch.cuda.is_available():
            print("Memory profiling only available for CUDA")
            return
        
        garment_img, garment_mask = self._generate_garment()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        checkpoints = []
        
        baseline = torch.cuda.memory_allocated() / (1024**2)
        checkpoints.append(('Baseline', baseline))
        
        garment_mesh = GarmentMesh.from_image(garment_img, garment_mask)
        garment_mem = torch.cuda.memory_allocated() / (1024**2)
        checkpoints.append(('Garment Generation', garment_mem))
        
        wrapped = self.wrapper.wrap_garment(garment_mesh, self.body_mesh, 'tshirt')
        wrap_mem = torch.cuda.memory_allocated() / (1024**2)
        checkpoints.append(('Mesh Wrapping', wrap_mem))
        
        _ = wrapped.render_to_image(np.eye(3, dtype=np.float32), (480, 640))
        render_mem = torch.cuda.memory_allocated() / (1024**2)
        checkpoints.append(('Rendering', render_mem))
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        checkpoints.append(('Peak Memory', peak_mem))
        
        print("\n| Stage | Memory (MB) | Delta (MB) |")
        print("|-------|-------------|------------|")
        
        prev_mem = baseline
        for name, mem in checkpoints:
            delta = mem - prev_mem
            print(f"| {name:25s} | {mem:10.1f} | {delta:+9.1f} |")
            prev_mem = mem
        
        print(f"\nTotal Memory Usage: {peak_mem:.1f} MB")
        print(f"RTX 2050 Capacity: 4096 MB")
        print(f"Remaining: {4096 - peak_mem:.1f} MB ({(1 - peak_mem/4096)*100:.1f}%)")
    
    def _generate_garment(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[50:200, 50:200] = [255, 0, 0]
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        return img, mask


def main():
    """Run all benchmarks and profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RTX 2050 Performance Benchmark')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to benchmark on')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations')
    parser.add_argument('--profile-memory', action='store_true',
                        help='Profile memory usage')
    parser.add_argument('--output', type=str, default='benchmarks/rtx2050_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    benchmark = PerformanceBenchmark(
        device=args.device,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )
    
    results = benchmark.run_all_benchmarks()
    benchmark.save_results(args.output)
    
    if args.profile_memory and args.device == 'cuda':
        profiler = MemoryProfiler(device=args.device)
        profiler.profile_memory_timeline()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
