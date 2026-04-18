"""
Chic India Complete System Orchestration
Launches and coordinates all services: Backend, Python ML, Database
"""
import subprocess
import time
import sys
import os
from pathlib import Path
import requests  # type: ignore

class ChicIndiaOrchestrator:
    """Manages all Chic India AR Platform services"""
    
    def __init__(self):
        self.processes = {}
        self.project_root = Path(__file__).parent
        
    def check_prerequisites(self):
        """Check if required software is installed"""
        print("\n" + "="*70)
        print("CHECKING PREREQUISITES")
        print("="*70)
        
        checks = {
            'Python': self._check_python(),
            'Node.js': self._check_node(),
            'PostgreSQL': self._check_postgres(),
            'Docker (optional)': self._check_docker()
        }
        
        for name, status in checks.items():
            icon = "✓" if status else "✗"
            print(f"{icon} {name}: {'Found' if status else 'NOT FOUND'}")
        
        if not all([checks['Python'], checks['Node.js']]):
            print("\n⚠ Missing required prerequisites!")
            sys.exit(1)
        
        print("\n✓ All prerequisites met")
        return True
    
    def _check_python(self):
        try:
            result = subprocess.run(['python', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_node(self):
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_postgres(self):
        try:
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_docker(self):
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def setup_database(self):
        """Initialize PostgreSQL database"""
        print("\n" + "="*70)
        print("DATABASE SETUP")
        print("="*70)
        
        # Check if .env exists in backend
        env_file = self.project_root / 'backend' / '.env'
        if not env_file.exists():
            print("Creating .env file...")
            example_env = self.project_root / 'backend' / '.env.example'
            if example_env.exists():
                import shutil
                shutil.copy(example_env, env_file)
                print("✓ Created .env from .env.example")
                print("⚠ Please update DATABASE_URL in backend/.env")
        
        # Try to start PostgreSQL with Docker if available
        if self._check_docker():
            print("\nStarting PostgreSQL with Docker...")
            try:
                subprocess.run([
                    'docker', 'run', '--name', 'chic-india-db',
                    '-e', 'POSTGRES_PASSWORD=secret',
                    '-p', '5432:5432',
                    '-d', 'postgres:15'
                ], check=False)
                print("✓ PostgreSQL container started")
                time.sleep(3)  # Wait for postgres to start
            except Exception as e:
                print(f"⚠ Docker setup failed: {e}")
        
        # Run Prisma migrations
        print("\nRunning database migrations...")
        backend_dir = self.project_root / 'backend'
        if backend_dir.exists():
            try:
                # Install dependencies first
                print("Installing backend dependencies...")
                subprocess.run(['npm', 'install'], 
                             cwd=backend_dir, check=True)
                
                # Generate Prisma client
                subprocess.run(['npx', 'prisma', 'generate'], 
                             cwd=backend_dir, check=True)
                
                # Run migrations
                subprocess.run(['npx', 'prisma', 'migrate', 'deploy'], 
                             cwd=backend_dir, check=False)
                
                print("✓ Database initialized")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Migration failed: {e}")
                print("You may need to run migrations manually:")
                print(f"  cd backend && npx prisma migrate dev")
    
    def start_python_ml_service(self):
        """Start Python ML service (FastAPI)"""
        print("\n" + "="*70)
        print("STARTING PYTHON ML SERVICE")
        print("="*70)
        
        # Install requirements
        print("Installing Python dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 
                       'fastapi', 'uvicorn', 'pydantic'], 
                      check=False)
        
        # Start FastAPI server
        print("Starting FastAPI server on port 8000...")
        process = subprocess.Popen(
            [sys.executable, 'python_ml_service.py'],
            cwd=self.project_root
        )
        self.processes['python_ml'] = process
        
        # Wait for service to be ready
        time.sleep(3)
        if self._check_service('http://localhost:8000/health'):
            print("✓ Python ML service started successfully")
        else:
            print("⚠ Python ML service may not be ready")
        
        return process
    
    def start_backend_api(self):
        """Start NestJS backend API"""
        print("\n" + "="*70)
        print("STARTING BACKEND API")
        print("="*70)
        
        backend_dir = self.project_root / 'backend'
        
        # Build TypeScript
        print("Building backend...")
        subprocess.run(['npm', 'run', 'build'], 
                      cwd=backend_dir, check=False)
        
        # Start server
        print("Starting NestJS server on port 3000...")
        process = subprocess.Popen(
            ['npm', 'run', 'start'],
            cwd=backend_dir
        )
        self.processes['backend'] = process
        
        # Wait for service to be ready
        time.sleep(5)
        if self._check_service('http://localhost:3000'):
            print("✓ Backend API started successfully")
        else:
            print("⚠ Backend API may not be ready")
        
        return process
    
    def _check_service(self, url, timeout=5):
        """Check if a service is responding"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def launch_demo(self):
        """Launch the Chic India AR demo"""
        print("\n" + "="*70)
        print("LAUNCHING AR DEMO")
        print("="*70)
        
        print("Starting live camera demo...")
        subprocess.run([sys.executable, 'launch_chic_india.py'],
                      cwd=self.project_root)
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n" + "="*70)
        print("STOPPING SERVICES")
        print("="*70)
        
        for name, process in self.processes.items():
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✓ All services stopped")
    
    def print_dashboard(self):
        """Print service dashboard"""
        print("\n" + "="*70)
        print("CHIC INDIA AR PLATFORM - SERVICE DASHBOARD")
        print("="*70)
        print("\nServices Running:")
        print("  ✓ Python ML Service:  http://localhost:8000")
        print("  ✓ Backend API:        http://localhost:3000")
        print("  ✓ PostgreSQL:         localhost:5432")
        print("\nEndpoints:")
        print("  • ML Service Docs:    http://localhost:8000/docs")
        print("  • Fit Prediction:     POST /predict-fit")
        print("  • Style Recommend:    POST /style-recommendations")
        print("  • Products API:       GET /products")
        print("  • Fit Prediction:     POST /fit-prediction")
        print("\nControls:")
        print("  • Press Ctrl+C to stop all services")
        print("="*70 + "\n")
    
    def run(self, mode='full'):
        """Run the complete system"""
        try:
            self.check_prerequisites()
            
            if mode in ['full', 'backend']:
                self.setup_database()
            
            if mode in ['full', 'ml']:
                self.start_python_ml_service()
            
            if mode in ['full', 'backend']:
                self.start_backend_api()
            
            if mode == 'full':
                self.print_dashboard()
                print("System ready! Press Ctrl+C to launch AR demo...")
                time.sleep(3)
                self.launch_demo()
            else:
                self.print_dashboard()
                print("Services running. Press Ctrl+C to stop...")
                # Keep running
                while True:
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.stop_all_services()
        
        except Exception as e:
            print(f"\n\nError: {e}")
            self.stop_all_services()
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chic India AR Platform Orchestrator')
    parser.add_argument('--mode', choices=['full', 'backend', 'ml', 'demo'], 
                       default='full', help='Launch mode')
    
    args = parser.parse_args()
    
    orchestrator = ChicIndiaOrchestrator()
    
    if args.mode == 'demo':
        orchestrator.launch_demo()
    else:
        orchestrator.run(mode=args.mode)


if __name__ == "__main__":
    main()
