#!/bin/bash
# bash train_to_acceptance.sh ./runcards/train.yml ./runcards/short_sample.yml

exit_gracefully () {
    rm ${run_id}/tmp
    echo "Exiting..."
    echo "Iterations completed: $n_iters"
    echo "Final train time: $((train_time / 60)) mins"
}
trap exit_gracefully EXIT
trap "echo ' Aborting!'; exit;" SIGINT

# Runcards
train_runcard=$1
sample_runcard=$2

# Parameters to be set by user
target_acceptance=0.7
n_sample=10

run_id=$(grep "training_output:" $sample_runcard | awk '{print $2}')
log_file=$run_id/training_log.out
data_file=$run_id/training_data.out

epochs=$(grep "epochs:" $train_runcard | awk '{print $2}')
n_iters=0
acceptance=0.0
train_time=0

IFS="
"

########## First run ############

start_time=$(date +%s)
anvil-train $train_runcard -o $run_id
end_time=$(date +%s)
((n_iters++))
((train_time+=(end_time-start_time) ))

######### Loop until acceptance ##########

while (( $(echo "$acceptance < $target_acceptance" | bc -l) ))
do
    
    start_time=$(date +%s)
    anvil-train $run_id -r -1 > ${run_id}/tmp # overwrite
    end_time=$(date +%s)
    ((train_time+=(end_time-start_time) ))
    
    for s in `seq 1 $n_sample`
    do
        anvil-sample $sample_runcard >> ${run_id}/tmp
    done
    echo "<<<<<<<<< ITERATION $n_iters >>>>>>>>>" >> $log_file
    cat ${run_id}/tmp >> $log_file

    ((n_iters++))
    echo "Completed $n_iters iterations"
    total_epochs=$(echo "$epochs * $n_iters" | bc -l)

    loss=$(grep "Final loss:" "${run_id}/tmp" | awk '{print $3}')
   
    arr=$(grep "Acceptance:" "${run_id}/tmp" | awk '{print $2}')
    sum=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    acceptance=$(echo "$sum / $n_sample" | bc -l)
    sum=$(echo $(for a in ${arr[@]}; do echo -n "($a - $acceptance)^2+"; done; echo -n 0) | bc -l)
    std_acceptance=$(echo "sqrt($sum / ($n_sample-1))" | bc -l)

    arr=$(grep "Integrated autocorrelation time:" "${run_id}/tmp" | awk '{print $4}')
    sum=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    tauint=$(echo "$sum / $n_sample" | bc -l)
    sum=$(echo $(for a in ${arr[@]}; do echo -n "($a-$tauint)^2+"; done; echo -n 0) | bc -l)
    std_tauint=$(echo "sqrt($sum / ($n_sample-1))" | bc -l)
    
    echo "$total_epochs $LOSS $acceptance $std_acceptance $tauint $std_tauint" >> "$data_file"

done

echo "Reached target acceptance"

