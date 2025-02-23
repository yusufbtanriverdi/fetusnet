# Run this bash for setup.
# TODO: Test across different OS.

eval "$(conda shell.bash hook)"

# Update Conda and create environment if needed
if ! conda info --envs | grep -q fetusnet; then
    echo "Creating conda environment 'fetusnet' .........................."
    conda create -y -n segment3d python=3.8
else
    echo "Conda environment 'fetusnet' already exists!"
fi

conda activate fetusnet 

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

echo "Environment is here!" 

# Get Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Check CUDA availability using Python
cuda_status=$(python - << EOF
import torch
print(torch.cuda.is_available())
EOF
)

# # Echo CUDA status
echo "$cuda_status"