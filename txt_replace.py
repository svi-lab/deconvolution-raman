#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:06:36 2019

@author: dejan
"""

# Read in the file
with open('Data/Etien/silica_600gf_1_profile-1.txt', 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('\t', '')

# Write the file out again
with open('Data/Etien/silica_600gf_1_profile-1.txt', 'w') as file:
  file.write(filedata)