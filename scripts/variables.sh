#!/usr/bin/env bash
#
# Created on Tue Jun 27 2023 13:40:14
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#

export XSOCK=/tmp/.X11-unix
export XAUTH=/tmp/.docker.xauth
export AVAILABLE_CORES=$(($(nproc) - 1))

export DOCKER_USER=theairlab
export IMAGE_NAME=wit-uas-dataset
export IMAGE_TAG=latest

export CONTAINER_NAME=$IMAGE_NAME
export CONTAINER_HOME_FOLDER=/root

HOST_UID=$(id -u)
HOST_GID=$(id -g)
export HOST_UID
export HOST_GID
