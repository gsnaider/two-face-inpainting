#!/bin/bash
for d in train/imgs/*/ ; do
	echo "${d}"
    IMGS=$(ls -l $d | wc -l)
    IMGS="$(($IMGS-1))"
    IDENTITY_DIR="$(basename "$d")"

    REAL_DIR="train/real/$IDENTITY_DIR"
    MASKED_DIR="train/masked/$IDENTITY_DIR"
    REFERENCE_DIR="train/reference/$IDENTITY_DIR"
    mkdir -p "${REAL_DIR}"
	mkdir -p "${MASKED_DIR}"
	mkdir -p "${REFERENCE_DIR}"

	REAL_COUNT="$(($IMGS*4/10))"
	mv $(ls -d "$PWD"/"${d}"/* | head -n "${REAL_COUNT}") "${REAL_DIR}"
	mv $(ls -d "$PWD"/"${d}"/* | head -n "${REAL_COUNT}") "${MASKED_DIR}"
	mv $(ls -d "$PWD"/"${d}"/*) "${REFERENCE_DIR}"

	rm -d "${d}"
done