#!/bin/bash
pip install kaggle
mkdir -p dataset
kaggle datasets download -d mokshbansal07/project-7 -p dataset --unzip
