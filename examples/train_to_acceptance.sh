#!/bin/bash
# bash train_to_acceptance.sh ./runcards/train.yml ./runcards/short_sample.yml

# --- Exit gracefully --- #
exit_gracefully () {
    rm ${run_id}/tmp
    echo "Exiting..."
    echo "Iterations completed: $n_iters ($epochs epochs)"
    echo "Current train time: $((train_time / 60)) mins"
}
files_not_found () {
    echo "Error: directory $run_id exists but no previous training data was found."
    echo "Please delete/rename $run_id before running this script again."
    exit
}
trap exit_gracefully EXIT
trap "echo ' Aborting!'; exit;" SIGINT

# --- Parameters to be set by user --- #
target_acceptance=0.7
n_sample=10

# --- Useful variables --- #
# Runcards from command line args
train_runcard=$1
sample_runcard=$2

run_id=$(grep "training_output:" $sample_runcard | awk '{print $2}')
log_file=$run_id/training_log.out
data_file=$run_id/training_data.out
meta_file=$run_id/ # to do

epochs_iter=$(grep "epochs:" $train_runcard | awk '{print $2}')
n_iters=0  # local

IFS="
"

# --- Initialise from existing data or start from scratch --- #
if [ -d $run_id ]; then
    [ ! -f $run_id/training_data.out ] && files_not_found
    epochs=$( tail -1 "${run_id}/training_data.out" | awk '{print $1}' )
    train_time=$( tail -1 "${run_id}/training_data.out" | awk '{print $2}' )
    acceptance=$( tail -1 "${run_id}/training_data.out" | awk '{print $4}' )
else
    epochs=0
    acceptance=0
    # Run an iteration without attempting sampling
    start_time=$(date +%s)
    anvil-train $train_runcard -o $run_id
    end_time=$(date +%s)
    ((n_iters++))
    train_time=$((end_time-start_time))
fi
echo "
      #################################################################
      #####                         NEW RUN                       #####
      #################################################################
     " >> $log_file

# --- Loop until target acceptance achieved --- #
while (( $(echo "$acceptance < $target_acceptance" | bc -l) ))
do
    # Training
    start_time=$(date +%s)
    anvil-train $run_id -r -1 > ${run_id}/tmp || exit # overwrite
    end_time=$(date +%s)
    ((train_time+=(end_time-start_time) ))
    
    # Run sampling n_sample times
    for s in `seq 1 $n_sample`
    do
        anvil-sample $sample_runcard >> ${run_id}/tmp
    done
    echo "<<<<<<<<< ITERATION $n_iters >>>>>>>>>" >> $log_file
    cat ${run_id}/tmp >> $log_file
    ((n_iters++))

    # Retrieve current state and write to log file
    echo "Completed $n_iters iterations"
    ((epochs+=epochs_iter))

    loss=$(grep "Final loss:" "${run_id}/tmp" | awk '{print $3}')
   
    arr=$(grep "Acceptance:" "${run_id}/tmp" | awk '{print $2}')
    sm=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    acceptance=$(echo "$sm / $n_sample" | bc -l)
    sm=$(echo $(for a in ${arr[@]}; do echo -n "($a - $acceptance)^2+"; done; echo -n 0) | bc -l)
    std_acceptance=$(echo "sqrt($sm / ($n_sample-1))" | bc -l)

    arr=$(grep "Integrated autocorrelation time:" "${run_id}/tmp" | awk '{print $4}')
    sm=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    tauint=$(echo "$sm / $n_sample" | bc -l)
    sm=$(echo $(for a in ${arr[@]}; do echo -n "($a-$tauint)^2+"; done; echo -n 0) | bc -l)
    std_tauint=$(echo "sqrt($sm / ($n_sample-1))" | bc -l)
    
    echo "$epochs $train_time $loss $acceptance $std_acceptance $tauint $std_tauint" >> "$data_file"

done

echo "Reached target acceptance"

