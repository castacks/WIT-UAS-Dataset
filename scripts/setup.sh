#!/usr/bin/env bash
#
# Created on Wed Jun 28 2023 01:56:49
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#

. "$(dirname "$0")"/variables.sh

sudo apt update

echo "Installing pre-commit"
pip3 install pre-commit

echo "Setting up pre-commit"
pre-commit install

echo "Done!"
