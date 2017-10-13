#!/usr/bin/env bash

DATAFILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/${DATAFILE}.tar.gz
TAR_DIR=./dataset/
mkdir ${TAR_DIR}
TAR_FILE=${TAR_DIR}/${DATAFILE}.tar.gz
wget -N ${URL} -O ${TAR_FILE}
tar -zxvf ${TAR_FILE} -C ./dataset/
rm ${TAR_FILE}
