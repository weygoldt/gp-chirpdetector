#!/usr/bin/env bash

color='\033[1;91m'
nocolor='\033[0m'
message='Running scripts in directory: '

for py_file in $(ls plot_*); do
    echo -e $message$color$py_file$nocolor
    python3 $py_file
done
