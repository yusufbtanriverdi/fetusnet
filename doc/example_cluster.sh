#!/bin/bash

# SLURM job configuration
#SBATCH -J msm                                    # Job name: 'msm'
#SBATCH -p high-cpu                                   # Partition (queue) name: high priority
#SBATCH --mem 32G                                # Request 32GB of memory
#SBATCH -N 1                                     # Request 1 node
#SBATCH -n 8                                     # Request 8 CPU cores
# #SBATCH --array=0-90:1%4                        # Run jobs 5-70 with step 1, max 4 concurrent jobs
#SBATCH -o /home/rgonzalezlopez/LOGS/surf_reg_neo/surf_%a.out         # Standard output file pattern (%a is array index)
#SBATCH -e /home/rgonzalezlopez/LOGS/surf_reg_neo/surf_%a.err         # Standard error file pattern

# Set up conda environment
export PATH="/home/rgonzalezlopez/miniconda3/bin:$PATH"  # Add conda to PATH
source activate surfenv                            # Activate 'surfenv' conda environment

# Load required software modules
module load FreeSurfer                           # Load FreeSurfer module

# Configure FSL environment
FSLDIR=/home/rgonzalezlopez/LIB/fsl                     # Set FSL directory
. ${FSLDIR}/etc/fslconf/fsl.sh                  # Source FSL configuration
PATH=${FSLDIR}/bin:${PATH}                      # Add FSL binaries to PATH
export FSLDIR PATH

# Configure Workbench environment
export PATH=$PATH:/home/rgonzalezlopez/LIB/workbench/bin_linux64                          # Add Workbench binaries to PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rgonzalezlopez/LIB/workbench/libs_linux64   # Add Workbench libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rgonzalezlopez/LIB/workbench/libs_linux64_software_opengl

# Dynamically retrieve subject folders from --input directory
input_dir="/home/rgonzalezlopez/DATA/DHCP_NEONATAL/rel3_dhcp_anat_pipeline_filt"
subject_dirs=($(ls -d $input_dir/sub-* | xargs -n 1 basename)) # Extract sub-* folder names
subject_ids=(${subject_dirs[@]#sub-})                         # Remove "sub-" prefix

# Get the current subject ID based on the SLURM array task ID
sub=${subject_ids[$SLURM_ARRAY_TASK_ID]}

python register_to_template.py \
    --input "$input_dir" \
    --i_derivative dhcp_measures_new \
    --output dhcp-surfreg-neonatal \
    --subject $sub \
    --config configs/config_strain.json