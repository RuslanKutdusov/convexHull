#!/bin/bash

./test
cd plots
./build_plots.sh
rm *.plot
cd ..
