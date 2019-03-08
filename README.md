Vehicle Detection for Cyclist Safety Training 

This repository contains code to facilitate the training of a deep neural network for vehicle detection.

In order to begin training, you must have a directory structure similar to the following:

+ training
  + data
    - train.record
    - val.record
  + models
    + model
      - pipeline.config

An example config for ssdlite_mobilenet_v2 is given in the home directory: `ssdlite_mobilenet_v2_vehicle_detection.config`. The commands to use to start training are given in `train.sh` which should be used as a guide (but may not work by simply running `. train.sh`.)

In order to create data, `generate_tfrecord.py` and `create_coco_tf_record.py` are provided. The former is made to use data from the `udacity self-driving car dataset` while the latter uses the relevent classes from `mscoco` (bicycle, car, motorcycle, bus, truck). In order to run the script with the correct inputs, `coco.sh` can be used as a guide.

In order to ensure that the correct images have been written to the tfrecord, `view_images.ipynb` can be used to view the images in a tfrecord.

Once training is complete, `freeze_model.sh` can be used to freeze the model for deployment.
