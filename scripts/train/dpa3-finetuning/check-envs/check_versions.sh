# Driver & GPUs (inside container with --gpus all)
nvidia-smi

# CUDA toolkit (only if nvcc is installed in the image)
nvcc --version || echo "nvcc not installed"

python -V
which python
python -m pip --version

python -m torch.utils.collect_env
python -m pip list | egrep -i 'deepmd|dpdata|ase|torch|numpy|scipy|pandas|matplotlib'


uname -a
cat /etc/os-release

# list branches (you already did)
dp --pt show ./DPA-3.1-3M.pt model-branch

# show type maps for every branch
dp --pt show ./DPA-3.1-3M.pt type-map

# show descriptor + fitting net + parameter sizes for every branch
dp --pt show ./DPA-3.1-3M.pt descriptor fitting-net size
