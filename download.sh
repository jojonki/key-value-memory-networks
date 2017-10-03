#!/bin/sh

DATA_DIR="./data"
mkdir -p $DATA_DIR

download () {
  URL=$1
  FILE_NAME=$2
  echo $URL
  echo $FILE_NAME
  if [ ! -f "$DATA_DIR$FILE_NAME.tar.gz" ]; then
    wget $URL$FILE_NAME.tar.gz -O "$DATA_DIR/$FILE_NAME.tar.gz"
  else
    echo "You've already downloaded $FILE_NAME dataset"
  fi

  if [ ! -f "$DATA_DIR/$FILE_NAME" ]; then
    tar zxvf "$DATA_DIR/$FILE_NAME.tar.gz" -C $DATA_DIR
  else
    echo "You've already unzipped $FILE_NAME dataset"
  fi
}


# 20 babi tasks
# Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin and Tomas Mikolov. 
# Towards AI Complete Question Answering: A Set of Prerequisite Toy Tasks, arXiv:1502.05698.
download "https://s3.amazonaws.com/fair-data/parlai/babi/" "babi/"

# movie dialog
# Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, Jason Weston. 
# Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems, arXiv:1511.06931.
# Warning: This contains invalid dataset. Use _pipe_ file instead of no pipe.
# Invalid value error because of labels data on Movie Dialog Dataset #313 (https://github.com/facebookresearch/ParlAI/issues/313)
download "https://s3.amazonaws.com/fair-data/parlai/moviedialog/" "moviedialog"

# WIKIMOVIES
# A. H. Miller, A. Fisch, J. Dodge, A. Karimi, A. Bordes, J. Weston. 
# Key-Value Memory Networks for Directly Reading Documents, arXiv:1606.03126.
# Warning: This contains invalid dataset. Use _pipe_ file instead of no pipe.
download "http://www.thespermwhale.com/jaseweston/babi/" "movieqa"