#!/bin/bash
for d in train/real/* ; do
	IDENTITY="$(basename "$d")"
	for f in "${d}"/* ; do
		IMG_FILE_NAME="$(basename "$f")"
		echo "${IDENTITY}/${IMG_FILE_NAME}" >> /tmp/real-files.txt
	done
done
shuf /tmp/real-files.txt > train/real/real-files.txt

for d in train/masked/* ; do
	IDENTITY="$(basename "$d")"
	for f in "${d}"/* ; do
		IMG_FILE_NAME="$(basename "$f")"
		echo "${IDENTITY}/${IMG_FILE_NAME}" >> /tmp/masked-files.txt
	done
done
shuf /tmp/masked-files.txt > train/masked/masked-files.txt