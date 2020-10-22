set -ex
conda_path=/shared/sami/conda
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
