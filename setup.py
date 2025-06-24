"""
This file can be used for the installation of this repository. 

Run with the --install flag to install the necessary requirements. 
If run without the --install flag, it will only compile the C++/CUDA code.

It is advised to run this script inside a virtual environment.
"""


import sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description="A script that process robot collected data to create mmWave image")
parser.add_argument("--install", action='store_true', help="If flag is provided, install necessary packages. Otherwise, only compile C++/CUDA code")
args = parser.parse_args()

# Only run install if requested 
if args.install:
    # Install main requirements
    proc = subprocess.Popen("pip install -r requirements.txt;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
    proc.wait()

    # Install additional requirements as needed
    from src.utils import utilities
    if utilities.load_param_json()['processing']['use_cuda']:
        proc = subprocess.Popen("pip install pycuda", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()

        # Note: You may need to change the pytorch version to match your CUDA version
        proc = subprocess.Popen("pip install torch --index-url https://download.pytorch.org/whl/cu115", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()

        proc = subprocess.Popen("pip install torchvision --index-url https://download.pytorch.org/whl/cu115", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()
    

        
# Build the C++/CUDA Code for normal estimation/imaging
from src.utils import utilities
proc = subprocess.Popen("make;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/data_processing/cpp")
proc.wait()
if utilities.load_param_json()['processing']['use_cuda']:
    proc = subprocess.Popen("nvcc --cubin -arch sm_86 --std=c++11 imaging_gpu.cu -o imaging_gpu.o;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/data_processing/cuda")
    proc.wait()
    proc = subprocess.Popen("nvcc --cubin -arch sm_86 --std=c++11 imaging_gpu.cu -o surface_normal_est.o;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/data_processing/cuda")
    proc.wait()

# Build the C++/CUDA code for isosurface optimization
if '--install' in sys.argv:
    sys.argv.remove('--install')
sys.argv.append("install")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='ray_tracing',
    ext_modules=[
        CUDAExtension('ray_tracing', [
            'src/data_processing/cpp/ray_tracing.cpp',
            'src/data_processing/cuda/ray_tracing_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
}
)
