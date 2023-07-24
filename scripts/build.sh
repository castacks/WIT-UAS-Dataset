#!/usr/bin/env bash
#
# Created on Tue Jun 27 2023 13:41:03
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

docker buildx build \
	--platform=linux/amd64 \
	--build-context home-folder-config="$(dirname "$0")"/../docker/build-context/home-folder-config \
	-t "$DOCKER_USER"/"$IMAGE_NAME":"$IMAGE_TAG" \
	- <"$(dirname "$0")"/../docker/"$IMAGE_TAG"/Dockerfile
