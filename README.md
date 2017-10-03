Color Segmentation
==================
### Requirements
  * OpenCV > 2.4.8 with Qt support


Color segmentation provides segmentation capability based on color information in HSV color space. The camera input to the node should be a `sensor_msgs/Image` topic. The identified segments are published under `color_segmentation/segments` topic.
The easiest way to run the node is using the launch file
```
roslaunch color_segmentation color_segmentation.launch

```
HSV limits
----------
The segmentation performance depends on finding the good thresholds for each image channel. When the node is run for the first time, it will run the OpenCV based control interface to setup the limits. This control screen can be displayed any time by setting `control_off` parameter to `True` while launching the node. When you are done with parameter setup, you can use `Save Config button` to save the configuration. Once a configuration is saved, the node will automatically read the thresholds and perform segmentation.


Workspace limits
----------------
The color segmentation node can work on the entire image. But there is an option to define a ROI for a specific portion of the image if the workspace of the robot is limited. By default it uses the workspace limits obtained from the [workspace_segmentation package](https://github.com/hkaraoguz/workspace_segmentation). However, you can create a custom file and supply the directory in the launch file. The file should be in the form:
```
min_x
max_x
min_y
max_y
```
All the parameters should be integers.
