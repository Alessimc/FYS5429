#!/bin/bash -f                      
#$ -N train_autoencoder                    # Set the job name
#$ -l h_rt=24:00:00                 # Set a hard runtime limit (hh:mm:ss)
#$ -S /bin/bash                     
#$ -pe shmem-1 1                    
#$ -l h_rss=32G,mem_free=32G,h_data=32G # Request memory 
#$ -q gpu-r8.q                  # Submit job to the 'gpu-r8.q' queue
#$ -l h=sm-nx10077659-bc-compute.int.met.no
#$ -j y                              # Merge standard output and error streams into a single file
#$ -m ba                             
#$ -o /lustre/storeB/users/alessioc/node_output/OUT_$JOB_NAME.$JOB_ID  # Standard output log file
#$ -e /lustre/storeB/users/alessioc/node_output/OUT_$JOB_NAME.$JOB_ID  # Standard error log file (merged with -j y)
#$ -R y                              
#$ -r y                              

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate FYS5429

# Define Python script
python_script="/lustre/storeB/users/alessioc/FYS5429/autoencoder/train_autoencoder.py"


# Execute the Python script with arguments
python3 $python_script --model 30 --batch_size 8 --nr_samples "all" --masked_loss True
