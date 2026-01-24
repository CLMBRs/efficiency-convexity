#!/bin/bash

# WCS Data
curl -O https://linguistics.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt
curl -O https://linguistics.berkeley.edu/wcs/data/20041016/WCS-Data-20110316.zip
unzip WCS-Data-20110316.zip
rm WCS-Data-20110316.zip

# IB Color Naming Model
curl -L -O https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1
unzip IB_color_naming_model.zip
mv IB_color_naming_model/IB_color_naming.pkl model.pkl
rm -r IB_color_naming_model
rm IB_color_naming_model.zip
