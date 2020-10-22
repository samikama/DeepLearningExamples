set -ex
conda_path=/shared/rejin/bin/conda
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
