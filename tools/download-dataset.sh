#!/usr/bin/env bash

DATAFILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/${DATAFILE}.tar.gz
TAR_FILE=./dataset/${DATAFILE}.tar.gz
TAR_DIR=./datset/${DATAFILE}/
wget -N ${URL} -O ${TAR_FILE}
mkdir ${TAR_DIR}
tar -zxvf ${TAR_FILE} -C ./dataset/
rm ${TAR_FILE}
