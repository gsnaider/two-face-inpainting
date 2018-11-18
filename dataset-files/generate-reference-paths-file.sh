#!/bin/bash
for d in train/reference/* ; do
	# echo "${d}"
	IDENTITY="$(basename "$d")"
	REFERENCES="${IDENTITY}:"
	for f in "${d}"/* ; do
		IMG_FILE_NAME="$(basename "$f")"
		REFERENCES="${REFERENCES}${IMG_FILE_NAME},"
	done
	# Remove the last ','
	echo "${REFERENCES::-1}" >> reference_files.txt
done