#!/bin/bash

python -u main.py \
  academy_3_vs_1_with_keeper \
  maac \
  --n_rollout_threads 4 \
  --n_controlled_lagents 3 \
  --n_controlled_ragents 0 \
  --buffer_length 1000000 \
  --n_episodes 50000 \
  --episode_length 100 \
  --steps_per_update 100 \
  --num_updates 4 \
  --batch_size 1024 \
  --pol_hidden_dim 512 \
  --critic_hidden_dim 512 \
  --attend_heads 4 \
  --save_interval 1000 \
  --pi_lr 0.005 \
  --q_lr 0.005 \
  --tau 0.005 \
  --gamma 0.99 \
  --reward_scale 10 \
  --reward_type checkpoints \
  --gpu \
  | tee train.log

#  --gpu \
#  --render \
