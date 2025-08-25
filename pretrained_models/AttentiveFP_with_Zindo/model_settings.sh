#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p XXXXXX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00    #1-00:00:00
#SBATCH -J train_fold
#SBATCH -o job.%u.%N.%j.out
#SBATCH --time=12:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
conda activate dml-minimal
export MPLBACKEND='Agg'

kfold_iter=10
main_dir=XXXXXX
save_dir=XXXXXX
run_name=XXXXXX
dataset_name=pre-processed_dataset_extra_features_wrdfs_9050_mols.pkl
fold_indices_file=10_kfold_indices_9050_mol.pkl
extra_feat_cols=[44,45,46,47]
mode="['gnn_and_zindo_only','None'][0]"
model_variant=AttentiveFP_zindo_A
epochs=400
batch_size=32
lr=0.0001
activ_str="nn.GELU()"
lr_cyc_start=120
add_residual="True"
gnn_cfg="{'gnn_in_channels': 45,         'gnn_hidden_channels': 256,         'gnn_out_channels': 256,         'gnn_num_layers': 4,         'gnn_num_timesteps': 4,         'gnn_edge_dim': 10,         'gnn_dropout':0.2,         'gnn_activation':nn.ReLU()}"
gnn_dense_cfg="{'input_dim':gnn_cfg['gnn_hidden_channels']+1,          'neurons': [256, 128, 32, 1],          'activations':[activ,activ,activ,None],          'dropout':[0.3, 0.3, 0.3, 0.0],          'use_BatchNorm':True}"
dense_zindo_cfg="None"
dense_fp_cfg="None"
dense_joint_cfg="None"
dense_final_cfg="None"


python_arguments=(
  "$kfold_iter"
  "$main_dir"
  "$save_dir"
  "$run_name"
  "$dataset_name"
  "$fold_indices_file"
  "$extra_feat_cols"
  "$mode"
  "$model_variant"
  "$epochs"
  "$batch_size"
  "$activ_str"
  "$lr"
  "$lr_cyc_start"
  "$add_residual"
  "$gnn_cfg"
  "$gnn_dense_cfg"
  "$dense_zindo_cfg"
  "$dense_fp_cfg"
  "$dense_joint_cfg"
  "$dense_final_cfg"
)


echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo -------------  
echo Job output begins                                           
echo -----------------                                           
echo

hostname

echo "---------------------------------"
echo " "
echo "Job id: ${SLURM_JOBID}."
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job submitted from:${SLURM_SUBMIT_DIR}"
echo " Using ${SLURM_JOB_NUM_NODES} node(s): ${SLURM_JOB_NODELIST}"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"
date
echo " "
echo "---------------------------------"
echo " "
echo "Done!"
date
echo " "
echo "---------------------------------"
# the ret flag is the return code, so you can spot easily if your code failed.
ret=$?

echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
exit $ret
