#!/usr/bin/env bash
#
# Created on Tue Jun 27 2023 13:40:31
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

docker pull "$DOCKER_USER"/"$IMAGE_NAME":"$IMAGE_TAG"
