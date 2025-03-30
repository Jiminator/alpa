#!/bin/bash
#SBATCH --mem=64G
#SBATCH --output="alpa.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS, 64 requests whole node
#SBATCH --partition=gpuA40x4    # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcrn-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=build-alpa
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -t 04:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

cd /scratch/bdkz/jshong/alpa
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate alpa
export HF_HOME=/scratch/bdkz/jshong/cache
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl 
module load cudnn/8.9.0.131
export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc
export TF_CUDA_PATHS=$CUDA_HOME

export CUDNN_HOME=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive
export CUDNN_INCLUDE_DIR=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/include
export CUDNN_LIB_DIR=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib
export CUDNN_PATH=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:$LD_LIBRARY_PATH
export CPATH=$CUDNN_INCLUDE_DIR:$CPATH
export LIBRARY_PATH=$CUDNN_LIB_DIR:$LIBRARY_PATH

export NCCL_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/nccl-2.19.3-1-464jrf3
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIB_DIR=$NCCL_HOME/lib
export TF_NCCL_VERSION=2.19.3
export TF_NCCL_PATHS=$NCCL_HOME

export CPATH=$NCCL_INCLUDE_DIR:$CPATH
export LIBRARY_PATH=$NCCL_LIB_DIR:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_LIB_DIR:$LD_LIBRARY_PATH

pip3 install --upgrade pip
pip3 install cupy-cuda11x
python3 -c "from cupy.cuda import nccl"
pip3 install -e ".[dev]"
cd build_jaxlib

time python3 build/build.py --enable_cuda --dev_install \
  --bazel_options="--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa \
  --override_repository=local_config_nccl=/scratch/bdkz/jshong/third_party/local_nccl \
  --action_env=CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc \
  --action_env=CUDA_TOOLKIT_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc \
  --action_env=CUDNN_INSTALL_PATH=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive \
  --action_env=CUDNN_INCLUDE_DIR=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/include \
  --action_env=CUDNN_LIB_DIR=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib \
  --action_env=CUDNN_HOME=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive \
  --action_env=TF_CUDA_PATHS=$CUDA_HOME \
  --action_env=TF_CUDA_VERSION=11.8 \
  --action_env=TF_CUDNN_VERSION=8 \
  --action_env=TF_NEED_CUDA=1 \
  --action_env=TF_CUDA_COMPUTE_CAPABILITIES=86 \
  --action_env=NCCL_HOME=$NCCL_HOME \
  --action_env=NCCL_INCLUDE_DIR=$NCCL_INCLUDE_DIR \
  --action_env=NCCL_LIB_DIR=$NCCL_LIB_DIR \
  --action_env=TF_NCCL_VERSION=2.19.3 \
  --action_env=TF_NCCL_PATHS=$NCCL_HOME \
  --action_env=PATH=$PATH \
  --action_env=LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  --action_env=CPATH=$CPATH \
  --action_env=LIBRARY_PATH=$LIBRARY_PATH"

cd dist
pip3 install -e .
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/nccl-2.19.3-1-464jrf3/lib:$LD_LIBRARY_PATH
ray start --head
# python3 -m alpa.test_install
cd /scratch/bdkz/jshong/alpa
python3 run_alpa.py
ray stop