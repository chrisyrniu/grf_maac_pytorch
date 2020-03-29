#!/bin/bash

python -u main.py \
  academy_3_vs_1_with_keeper \
  maac \
  --n_rollout_threads 1 \
  --n_controlled_lagents 3 \
  --n_controlled_ragents 0 \
  --buffer_length 1000000 \
  --n_episodes 20 \
  --episode_length 200 \
  --steps_per_update 100 \
  --num_updates 4 \
  --batch_size 1024 \
  --pol_hidden_dim 128 \
  --critic_hidden_dim 128 \
  --attend_heads 4 \
  --save_interval 500 \
  --pi_lr 0.001 \
  --q_lr 0.001 \
  --tau 0.001 \
  --gamma 0.99 \
  --reward_scale 100 \
  --reward_type checkpoints \
  | tee train.log

#  --gpu \
#  --render \
