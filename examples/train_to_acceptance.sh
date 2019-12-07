#!/bin/bash

trap "echo ' Aborting...'; exit;" SIGINT

# Runcards
TRAIN_RUNCARD=$1
SAMPLE_RUNCARD=$2

# Parameters to be set by user
TARGET_ACCEPTANCE=0.7
N_SAMPLE=10


RUN_ID=$(grep "training_output:" $SAMPLE_RUNCARD | awk '{print $2}')
LOG_FILE=$RUN_ID/training_log.out
DATA_FILE=$RUN_ID/training_data.out

EPOCHS=$(grep "epochs:" $TRAIN_RUNCARD | awk '{print $2}')
N_ITERS=0
ACCEPTANCE=0.0

IFS="
"

########## First run ############

anvil-train $TRAIN_RUNCARD -o $RUN_ID
((N_ITERS++))

######### Loop until acceptance ##########

while (( $(echo "$ACCEPTANCE < $TARGET_ACCEPTANCE" | bc -l) ))
do
    
    anvil-train $RUN_ID -r -1 > "$LOG_FILE" # overwrite
    
    for s in `seq 1 $N_SAMPLE`
    do
        anvil-sample $SAMPLE_RUNCARD >> "$LOG_FILE"
    done

    ((N_ITERS++))
    echo "Completed $N_ITERS iterations"
    TOTAL_EPOCHS=$(echo "$EPOCHS * $N_ITERS" | bc -l)

    LOSS=$(grep "Final loss:" "$LOG_FILE" | awk '{print $3}')
   
    arr=$(grep "Acceptance:" "$LOG_FILE" | awk '{print $2}')
    sum=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    ACCEPTANCE=$(echo "$sum / $N_SAMPLE" | bc -l)
    sum=$(echo $(for a in ${arr[@]}; do echo -n "($a - $ACCEPTANCE)^2+"; done; echo -n 0) | bc -l)
    STD_ACCEPTANCE=$(echo "sqrt($sum / ($N_SAMPLE-1))" | bc -l)

    arr=$(grep "Integrated autocorrelation time:" "$LOG_FILE" | awk '{print $4}')
    sum=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    TAUINT=$(echo "$sum / $N_SAMPLE" | bc -l)
    sum=$(echo $(for a in ${arr[@]}; do echo -n "($a-$TAUINT)^2+"; done; echo -n 0) | bc -l)
    STD_TAUINT=$(echo "sqrt($sum / ($N_SAMPLE-1))" | bc -l)
    
    echo "$TOTAL_EPOCHS $LOSS $ACCEPTANCE $STD_ACCEPTANCE $TAUINT $STD_TAUINT" >> "$DATA_FILE"

done

echo "Reached target acceptance"

