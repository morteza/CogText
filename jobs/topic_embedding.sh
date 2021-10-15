#!/bin/bash

#SBATCH --job-name=cogtext_topic_embedding
#SBATCH --chdir=/work/projects/acnets/repositories/
#SBATCH --output=/work/projects/acnets/logs/%x_%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


echo "SLURM_JOB_ID: " $SLURM_JOB_ID


# enable access to the `module` and install conda.
[ -f /etc/profile ] && source /etc/profile
module purge
module load lang/Anaconda3

# purge old stuff (conda env, codes, data, outputs, etc)
conda deactivate
conda env remove -n cogtext

if [ ! -d "cogtext/" ]
then 
  git clone --recurse-submodules ssh://git@gitlab.uni.lu:8022/xcit/efo/cognitive-tests-text-analysis.git cogtext/
  cd cogtext/
else
  cd cogtext/
  git reset --hard
  git clean -xdf
  git pull --recurse-submodules
fi


# create and prepare a new conda environment (also installs git and git-lfs via conda)
conda create -n cogtext python=3.9 --yes
conda activate cogtext
conda update conda
conda install -c conda-forge git git-lfs -y
conda install -c conda-forge cudatoolkit -y

# install dependencies and ipython to run the notbooks
pip install pip -U
pip install -r requirements_hpc.txt
# pip install ipython jupyter

# to avoid memory limit issue in HDBSCAN
pip install --upgrade git+https://github.com/scikit-learn-contrib/hdbscan

# Fix transformers bug when nprocess > 1
export TOKENIZERS_PARALLELISM=false

# run the code
python jobs/topic_embedding.py -f 1.0 --bertopic --top2vec

# push the BIDS changes back to the gitlab repository
git lfs install
git add -A .
git commit -m "CI/HPC/topic_embedding.sh auto-commit (SLURM_JOB_ID: ${SLURM_JOB_ID})"
git push origin dev

# That's it!
