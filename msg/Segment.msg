int32 pixelposcenterx  # x axis pixel position of the segment center
int32 pixelposcentery  # y axis pixel position of the segment center
sensor_msgs/Image image # image of the segment
geometry_msgs/Point32[] pixelhull # the convex hull of the segment in pixel coordinates
int8 averagehue # the average hue value of the segment
