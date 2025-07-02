##### SBATCH PARAMETERS ##### 

#!/bin/bash
# SBATCH -J herewegoagain
# SBATCH -p high
# SBATCH -N 1
#SBATCH -n 8                                     # Request 8 CPU cores
# SBATCH --nodelist=node032
# SBATCH --chdir=/home/jreyes
# SBATCH --mem=32GB 
# SBATCH --gres=gpu:gtx1080:1
# #SBATCH --array=0-10:1%4                        # Run 10 jobs with step 1, max 4 concurrent jobs

#SBATCH -o /home/ytanriverdi/%J.%u.out # STDOUT
#SBATCH -e /home/ytanriverdi/%J.%u.err # STDERR 

## SBATCH --partition=<partition>          # Partition/queue name
## SBATCH --nodes=<num_nodes>              # Number of nodes to use
## SBATCH --ntasks=<num_tasks>             # Number of tasks (can differ from nodes)
## SBATCH --cpus-per-task=<num_cpus>       # Number of CPUs per task
## SBATCH --gres=gpu:<type>:<num_gpus>     # Request GPUs (type and number)
## SBATCH --time=<time>                    # Maximum job time (format D-HH:MM:SS)
## SBATCH --mem=<memory>                   # Memory per node
## SBATCH --mail-user=<email>              # Email for notifications
## SBATCH --mail-type=BEGIN,END,FAIL       # Notification types: start, end, fail
## SBATCH --output=<path_to_stdout>        # Path for standard output (STDOUT)
## SBATCH --error=<path_to_stderr>         # Path for error output (STDERR)
## SBATCH --nodelist=<node_list>           # Specific nodes to use
## SBATCH --exclude=<node_list>            # Nodes to exclude
## SBATCH --chdir=<directory>              # Working directory
## SBATCH --dependency=<job_id>            # Start after a specific job
## SBATCH --requeue                        # Requeue job if it fails
## SBATCH --array=<index_list>             # Run an array of jobs
## SBATCH --account=<account>              # Billing account (if required)
## SBATCH --constraint=<features>          # Restrict to nodes with specific features

##### LOAD CUDA and ANACONDA #####

# Load CUDA module
module load CUDA/12.1
nvcc --version

# Load Anaconda module
module load Anaconda3/2020.02

##### SET UP THE ENV #####
# Run this bash for setup.
# TODO: Test across different OS.

eval "$(conda shell.bash hook)"

# Update Conda and create environment if needed
if ! conda info --envs | grep -q fetusnet; then
    echo "Creating conda environment 'fetusnet' .........................."
    conda create -y -n fetusnet python=3.8
else
    echo "Conda environment 'fetusnet' already exists!"
fi

conda activate fetusnet 

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r doc/requirements/requirements.txt

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

### JOBS 

python fetusnet.py \
    # --input "$input_dir" \
    # --i_derivative dhcp_measures_new \
    # --output dhcp-surfreg-neonatal \
    # --subject $sub \
    # --config configs/config_strain.json
    --params ex1_params.json
    --iter_folds $sub1
    --dataset $sub2
    --prefix cluster