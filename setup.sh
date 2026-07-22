#! /bin/bash
# Purpose: A shell script to run the initial setup of the project in a single command
# Usage: 
#    . setup.sh                --> Create the enviroment, loading from existing environment if exists
#    . setup.sh recreate       --> Recreate an environment from scratch and overwrite cache
#    . setup.sh recreate_temp  --> Recreate an environment from scratch but do not overwrite existing cache


eval "$(conda shell.bash hook)"

git config --local include.path ../.gitconfig

# default empty value
if [ -z "$1" ]; then 
    conda activate base
    if ! make conda_check_cache; then
    echo "Cache does not exist on S3."
    echo ">>>  Setting up environment and caching for future use"
    make environment &&
    conda activate onesnapshot-nlp &&
    make requirements &&
    make conda_cache_s3 &&
    echo -e ">>> New conda env created. Activate with: \n conda activate onesnapshot-nlp"
    else
	echo ">>> Cached environment detected - recreating."
    make get_conda_cache &&
    conda activate onesnapshot-nlp &&
    python -m ipykernel install --user --name=onesnapshot-nlp &&
	echo -e ">>> Cached conda env re-created. Activate with:\nconda activate onesnapshot-nlp"
    fi
    
    
elif [ "$1" == "recreate_temp" ]; then
    echo "recreating conda environment and installing packages (without caching for future use)"
    conda activate base &&
    make environment &&
    conda activate onesnapshot-nlp &&
    make requirements

elif [ "$1" == "recreate" ]; then
    echo "loading cached conda environment, installing packages and overwriting cache for future use"
    conda activate base &&
    rm -f repo_settings/conda.lock.yml &&
    make environment &&
    conda activate onesnapshot-nlp &&
    make requirements &&
    make conda_cache_s3 &&
    conda env export --from-history --name onesnapshot-nlp > repo_settings/conda.lock.yml

# running setup.sh will always read or create a conda env
else
    echo "incorrect flag:" $1
    echo "correct usage:"
    echo ". setup.sh           --> Create the enviroment, loading from cached environment if exists"
    echo ". setup.sh recreate  --> Recreate an environment from scratch and overwrite cache"
    echo ". setup.sh recreate_temp  --> Recreate an environment from scratch but do not overwrite existing cache"
    
fi

conda activate onesnapshot-nlp &&
echo conda env activated!
