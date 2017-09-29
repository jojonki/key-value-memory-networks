#!/bin/sh

DATA_DIR="./data"
FILE_NAME="moviedialog.tar.gz"
UNZ_FILE_NAME="moviedialog"

mkdir -p $DATA_DIR

if [ ! -f "$DATA_DIR/$FILE_NAME" ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/moviedialog/moviedialog.tar.gz -O "$DATA_DIR/$FILE_NAME"
else
  echo "You've already downloaded dataset"
fi

if [ ! -f "$DATA_DIR/$UNZ_FILE_NAME" ]; then
  tar zxvf "$DATA_DIR/$FILE_NAME" -C $DATA_DIR
else
  echo "You've already unzipped dataset"
fi


