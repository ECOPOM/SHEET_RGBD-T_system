#include <ros/ros.h>
#include <sensor_msgs/Image.h>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "seek.h"
#include <iostream>
using namespace cv;
using namespace LibSeek;


class Thermal_camera{
    private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    //Parameters
    int smoothing;
    int warmup;
    std::string camtype;
    int colormap;
    int rotate;

    //Seek variables
    int i;
    cv::Mat frame_u16, frame, outframe, outframe_norm;
    LibSeek::SeekThermalPro seekpro;
    LibSeek::SeekThermal seek;
    LibSeek::SeekCam* cam;

    int state = 0;

    //ROS Publishers
    ros::Publisher img_pub;
    ros::Publisher img_pub_norm;

    int open_camera();
    int acquire_img();
    void pub_image();

    public:
    double spin_rate;

    Thermal_camera();
    int management();
};