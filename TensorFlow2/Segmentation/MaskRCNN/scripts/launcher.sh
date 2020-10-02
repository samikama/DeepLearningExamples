set -ex
conda_path=/shared/conda
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
