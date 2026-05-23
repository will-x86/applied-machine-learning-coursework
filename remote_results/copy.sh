#!/usr/bin/env sh
set -e

scp -r pop-os:/home/will/projects/applied-machine-learning-coursework/comparison_confusion.png "$1.png"
scp -r pop-os:/home/will/projects/applied-machine-learning-coursework/results.txt "$1.txt"

#TYPE=$1
#
#if [ -n "$2" ]; then
#    N=$2
#    DIR="${TYPE}_${N}"
#    echo "Overwriting $DIR..."
#    #rm -rf "$DIR"
#else
#    N=1
#    while [ -d "${TYPE}_${N}" ]; do
#        N=$((N + 1))
#    done
#    echo "Creating $DIR..."
#fi
#
#DIR="${TYPE}_${N}"
#CODE_DIR="${DIR}/code/"
#
#mkdir -p "$CODE_DIR"
#echo "Copying code from ../code/$TYPE/..."
#cp -r "../code/$TYPE/." "$CODE_DIR/"
#
#cd "$DIR"
#echo "Pulling logs..."
#scp -r pop-os:/home/will/projects/aiab/code/logs/ .
#echo "Done. You're now in: $DIR"
#
