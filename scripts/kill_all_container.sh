#!/usr/bin/env bash
#
# Created on Tue Jun 27 2023 15:30:34
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#

. "$(dirname "$0")"/variables.sh

echo "Removing all containers"
docker rm -f "$(docker ps -aq)"
echo "Done"
