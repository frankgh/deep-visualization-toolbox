#!/usr/bin/env bash

# Exit on first error
set -e

function fetch_file() {
    filename="$1"
    url="$2"
    if [ -e "$filename" ]; then
        echo "$url: file already downloaded (remove $filename to force re-download)"
    else
        echo "$url: fetching..."
        wget -O "$filename" "$url"
        echo "$url: done."
    fi
}

function fetch_and_extract() {
    filename="$1"
    url="$2"
    dir="$3"
    example_filename="$4"
    example_path="$dir/$example_filename"
    if [ -e "$example_path" ]; then
        echo "$url: $example_path already exists, skipping."
    else
        fetch_file "$filename" "$url"
        echo "$url: extracting..."
        mkdir -p "$dir"
        tar -C "$dir" -xzf "$filename"
        echo "$url: done."
    fi
}


fetch_and_extract model.tar.gz "https://firebasestorage.googleapis.com/v0/b/frankgh-com.appspot.com/o/kerasvis-afg%2Fmodel.tar.gz?alt=media&token=5d279b91-0962-486d-a8dd-1460507dedaf" . model.h5

fetch_and_extract patients-small.tar.gz "https://firebasestorage.googleapis.com/v0/b/frankgh-com.appspot.com/o/kerasvis-afg%2Fpatients-small.tar.gz?alt=media&token=4cdb6c60-b0a3-44e2-9c54-b20abd50b018" patients

if [ "$1" = "all" ]; then
    fetch_and_extract patients.tar.gz "https://firebasestorage.googleapis.com/v0/b/frankgh-com.appspot.com/o/kerasvis-afg%2Fpatients.tar.gz?alt=media&token=de700049-747a-43a8-879e-b5e3855efcf5" patients
else
    echo
    echo "Rerun as \"$0 all\" to also fetch all patients (Warning: 3.5G more)"
fi
