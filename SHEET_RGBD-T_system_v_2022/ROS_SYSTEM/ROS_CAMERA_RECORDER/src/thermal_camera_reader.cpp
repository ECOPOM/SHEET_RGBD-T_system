#include <thermal_camera_reader/thermal_camera_reader.hpp>
//#include <SeekThermalPro.h>


Thermal_camera::Thermal_camera():
    nh_(ros::NodeHandle()),
    nh_private_(ros::NodeHandle("~"))
{
    img_pub = nh_.advertise<sensor_msgs::Image>("/thermal_img", 1, this);
    img_pub_norm = nh_.advertise<sensor_msgs::Image>("/thermal_img_normalized", 1, this);
    nh_private_.param<int>("smoothing", smoothing, 1);
    nh_private_.param<int>("warmup", warmup, 10);
    nh_private_.param<std::string>("cam_type", camtype, "seekpro");
    nh_private_.param<int>("color_map", colormap, -1);
    nh_private_.param<int>("rotate", rotate, 270);
    nh_private_.param<double>("spin_rate", spin_rate, 10.0);
}

int Thermal_camera::open_camera()
{
    // Init correct cam type
    if (camtype == "seekpro") {
        cam = &seekpro;
    }
    else {
        cam = &seek;
    }

    if (!cam->open()) {
        ROS_ERROR_STREAM("failed to open " << camtype << " cam");
        return -1;
    }

    // Warmup
    for (i = 0; i < warmup; i++) {
        if (!cam->grab()) {
            ROS_ERROR_STREAM("no more LWIR img");
            return -1;
        }
        cam->retrieve(frame_u16);
        //cv::waitKey(10);
    }

    ROS_INFO_STREAM("warmup complete");
    return 0;
}

int Thermal_camera::acquire_img()
{
    Mat avg_frame;

    // Aquire frames
    for (i = 0; i < smoothing; i++) {
        if (!cam->grab()) {
            ROS_ERROR_STREAM("no more LWIR img");
            return -1;
        }

        cam->retrieve(frame_u16);
        frame_u16.convertTo(frame, CV_32FC1);

        if (avg_frame.rows == 0) {
            frame.copyTo(avg_frame);
        } else {
            avg_frame += frame;
        }
    }

    // Average the collected frames
    avg_frame /= smoothing;
    avg_frame.convertTo(frame_u16, CV_16UC1);

    Mat frame_g8; // Transient Mat containers for processing

    outframe_norm = frame_u16.clone();

    normalize(outframe_norm, outframe_norm, 0, 65535, NORM_MINMAX);

    // Convert seek CV_16UC1 to CV_8UC1
    outframe_norm.convertTo(outframe_norm, CV_8UC1, 1.0 / 256.0);

    //cv::imshow("frame_g8", frame_g8);
    //cv::waitKey();
    // Apply colormap: https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
    //if (colormap != -1) {
    //    applyColorMap(frame_g8, outframe, colormap);
    //}
    //else {
    //    cv::cvtColor(frame_g8, outframe, cv::COLOR_GRAY2BGR);
    //}
    outframe = frame_u16.clone();

    // Rotate image
    if (rotate == 90) {
        transpose(outframe, outframe);
        flip(outframe, outframe, 1);
        transpose(outframe_norm, outframe_norm);
        flip(outframe_norm, outframe_norm, 1);
    }
    else if (rotate == 180) {
        flip(outframe, outframe, -1);
        flip(outframe_norm, outframe_norm, -1);
    }
    else if (rotate == 270) {
        transpose(outframe, outframe);
        flip(outframe, outframe, 0);
        transpose(outframe_norm, outframe_norm);
        flip(outframe_norm, outframe_norm, 0);
    }

    //cv::imshow("rotated", outframe);
    //cv::waitKey();
    pub_image();
    return 0;
}

void Thermal_camera::pub_image()
{
    sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(),"mono16", outframe).toImageMsg();
    img_pub.publish(msg1);

    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(),"mono8", outframe_norm).toImageMsg();
    img_pub_norm.publish(msg2);

}

int Thermal_camera::management()
{
    switch(state){
        case 0://start
            if(!open_camera())
                state = 10;
            else
                state = 100;
            break;

        case 10://image acquisition
            if(acquire_img())
                state = 100;
            break;
                
        case 100://error
            return 1;
    }
    return 0;
}