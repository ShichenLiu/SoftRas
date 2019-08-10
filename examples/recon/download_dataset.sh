#!/usr/bin/env bash

if [ -e "./data/mesh_reconstruction_dataset.zip" ]; then
    unzip ./data/mesh_reconstruction_dataset.zip -d ./data/datasets
    mv ./data/datasets/mesh_reconstruction/* ./data/datasets/
    rm -rf /data/datasets/mesh_reconstruction
else
    echo "Please download dataset from https://drive.google.com/open?id=1fY9IWK7yEfLOmS3wUgeXM3NIivhoGhsg, put it in ./data/mesh_reconstruction_dataset.zip, and run this script."
fi