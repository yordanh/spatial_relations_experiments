#!/bin/bash
set -e

seeds="111 222 333 444 555 123 234 345 456 567"

# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "euclid" --state_type "embed" &
# 	sleep 2m 10s
# 	kill -9 $!
# done



# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "euclid_dist" --state_type "eef" &
# 	sleep 30s
# 	kill -9 $!
# done



# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "euclid" --state_type "eef" &
# 	sleep 35s
# 	kill -9 $!
# done



# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "discrete" --state_type "eef" &
# 	sleep 30s
# 	kill -9 $!
# done



for i in $seeds
do
	python ./src/train_trpo_gym_master.py --seed $i --reward_type "discrete" --state_type "embed" &
	sleep 2m 10s
	kill -9 $!
done



# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "embed" --state_type "eef" &
# 	sleep 2m 10s
# 	kill -9 $!
# done




# for i in $seeds
# do
# 	python ./src/train_trpo_gym_master.py --seed $i --reward_type "embed" --state_type "embed" &
# 	sleep 3m 30s
# 	kill -9 $!
# done