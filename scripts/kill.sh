#!/bin/sh
#
# Created on Tue Jun 27 2023 13:40:37
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

docker exec --privileged -it "$CONTAINER_NAME" pkill -f ros
docker exec --privileged -it "$CONTAINER_NAME" pkill -f rviz
docker exec --privileged -it "$CONTAINER_NAME" pkill -f zsh
pkill -f ros
pkill -f rviz

tmux kill-session -t Tartan-SLAM

docker rm -f "$CONTAINER_NAME"
