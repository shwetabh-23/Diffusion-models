#!/bin/bash

kaggle datasets download -d 'joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution'
unzip 'cifar10-64x64-resized-via-cai-super-resolution.zip'

mv cifar10-64 data
