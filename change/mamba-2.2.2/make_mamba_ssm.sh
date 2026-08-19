export CC=gcc-11
export CXX=g++-11
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
pip install -e . --no-deps --no-build-isolation
python -c "from mamba_ssm import Mamba; print('✅ Mamba imported successfully')"