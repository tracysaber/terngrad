#!/bin/bash
set -x
set -e
PS=192.168.255.87
WORKER1=192.168.255.87
WORKER2=192.168.255.91
TRAIN_DIR=/tmp/cifar/1
NET=cifar10_alexnet
#export CUDA_VISIBLE_DEVICES=1
#bazel-bin/inception/cifar10_eval \
#--eval_interval_secs  10 \
#--device "/gpu:0" \
#--restore_avg_var True  \
#--data_dir ${HOME}/dataset/cifar10-data \
#--subset "test" \
#--net ${NET} \
#--image_size 24 \
#--batch_size 50 \
#--max_steps 10000 \
#--checkpoint_dir ${TRAIN_DIR} \
#--tower 0 \
#--eval_dir /tmp/cifar/1_eval &

bazel-bin/inception/cifar10_distributed_train \
--grad_bits 1 \
--clip_factor 2.5 \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 128 \
--num_epochs_per_decay 200 \
--max_steps 30000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--save_iter 500 \
--image_size 24 \
--data_dir="$HOME/dataset/cifar10-data-shard-500-999" \
--job_name='worker' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2222" \
--train_dir=/tmp/cifar/1 &

#export CUDA_VISIBLE_DEVICES=1
bazel-bin/inception/cifar10_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2222"
