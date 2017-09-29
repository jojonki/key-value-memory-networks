#!/bin/sh

DATA_DIR="./data"
BABI_FILE_NAME="babi"
MOV_FILE_NAME="moviedialog"

mkdir -p $DATA_DIR

# babi 20 tasks
if [ ! -f "$DATA_DIR/$BABI_FILE_NAME.tar.gz" ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/babi/$BABI_FILE_NAME.tar.gz -O "$DATA_DIR/$BABI_FILE_NAME.tar.gz"
else
  echo "You've already downloaded babi dataset"
fi

if [ ! -f "$DATA_DIR/$BABI_FILE_NAME" ]; then
  tar zxvf "$DATA_DIR/$BABI_FILE_NAME.tar.gz" -C $DATA_DIR
else
  echo "You've already unzipped babi dataset"
fi


# movie dialog data
if [ ! -f "$DATA_DIR/$MOV_FILE_NAME.tar.gz" ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/moviedialog/$MOV_FILE_NAME.tar.gz -O "$DATA_DIR/$MOV_FILE_NAME.tar.gz"
else
  echo "You've already downloaded moviedialog dataset"
fi

if [ ! -f "$DATA_DIR/$MOV_FILE_NAME" ]; then
  tar zxvf "$DATA_DIR/$MOV_FILE_NAME.tar.gz" -C $DATA_DIR
else
  echo "You've already unzipped moviedialog dataset"
fi


