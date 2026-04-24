#!/bin/bash
today=$(date +"%Y-%m-%d")
logdir="logs/$today"
mkdir -p "$logdir"

# BIG_DATA_PATH="/home/asim.ukaye/fed_learning/datasets/"
# BIG_DATA_PATH="/l/users/asim.ukaye/"
# Define your argument list here

splits=(
    # iid
    # dirichlet_0.1
    # dirichlet_0.01
    # step_quantity
    # step_label_skew
    only_label_skew
)

# splitsA=(
#     iid
#     dirichlet_0.01
#     only_label_skew

# )
# splitsB=(
#     # dirichlet_0.01
#     step_quantity
#     step_label_skew
# )
# # splitsC=(
# #     only_label_skew
# # )
# splits=("${splitsA[@]}" "${splitsB[@]}")

mus=(0.0 0.1 0.3 0.5 0.7 0.9)

# seeds=(40 41 43 44)
# seeds=(42 43)
seeds=(41)

# strategy="spectralfedopt"
# strategy="spectralfed"
# strategy="standalone"
# strategy="shapfed"    
strategy="fedavg_uni"
# strategy="cgsv"
# strategy="fedopt_uni"

# dataset="mnist"
# model="mlpnet"


# dataset="cifar10"
# model="tf_cnn"

dataset="cifar100"
model="resnet50"
# seed=41

reward="none"
rewards=False
suffix="_new_seed_$seed"

# suffix="_new_mu0.9_seed_$seed"
# suffix="no_new"
# suffix="interp_mu0.5_new"
mu=0.9
# reward="interpolation"
# reward="sparsification
# split="dirichlet_1.0"
# split="dirichlet_0.1"
# split="iid"
# Loop over the arguments and launch jobs
for split in "${splits[@]}"; do
for seed in "${seeds[@]}"; do
# for splitset in "${splits[@]}"; do
# for split in $splitset; do
    suffix="_new_seed_$seed"

    timestamp=$(date +"%H%M%S%N")
    logfile="$logdir/${timestamp}_${strategy}_${split}.log"
    # echo "Starting process for $strategy with argument: $split"
    echo "Starting process for $strategy , $split with seed: $seed"
    if [[ "$strategy" = "spectralfed"* ]]; then
        python spectralfed.py --strategy $strategy --split "$split" --dataset $dataset --model $model --mu $mu --seed $seed --reward $reward --suffix "$suffix" > "$logfile" 2>&1 &
    elif [[ "$strategy" = "shapfed"* ]]; then
        python shapfed.py --split "$split" --dataset $dataset --model $model  --suffix "$suffix" --rewards $rewards --seed $seed > "$logfile" 2>&1 &
    elif [[ "$strategy" = "standalone"* ]]; then
        python standalone.py --split "$split" --dataset "$dataset" --model "$model" --suffix "$suffix" --seed $seed > "$logfile" 2>&1 &
    elif [[ "$strategy" = "fedavg"* ]] || [[ "$strategy" = "fedopt"* ]]; then
        python fedavg.py --strategy "$strategy" --split "$split" --dataset "$dataset" --model "$model" --suffix "$suffix" --seed $seed > "$logfile" 2>&1 &
    elif [[ "$strategy" = "cgsv"* ]]; then
        python cgsv.py --strategy "$strategy" --split "$split" --dataset "$dataset" --model "$model" --suffix "$suffix" --seed $seed > "$logfile" 2>&1 &
    fi
    sleep 2
    done
    wait
done

# python spectralfed.py --split "only_label_skew" --dataset "cifar10" --model "tf_cnn" --reward "sparsification" --suffix "rwd_sparsification"

# for mu in "${mus[@]}"; do
#     timestamp=$(date +"%y%m%d_%H%M%S%N")
#     logfile="logs/${timestamp}_${dataset}_mu${mu}.log"
#     echo "Starting process with mu: $mu"
#     # python fedavg.py --strategy fedavg_uni --split "$arg" > "$logfile" 2>&1 &
#     # python spectralfed.py --strategy $strategy --dataset cifar10 --model tf_cnn --split step_label_skew --mu $mu --suffix "mu=$mu"> "$logfile" 2>&1 &
#     python spectralfed.py --strategy $strategy --dataset $dataset --model $model --split $split --mu $mu --suffix "mu=$mu"> "$logfile" 2>&1 &
#     # python fedavg.py --split "$arg" > "$logfile" 2>&1 &
# done

# Wait for all background jobs to complete
wait
echo "All processes completed."