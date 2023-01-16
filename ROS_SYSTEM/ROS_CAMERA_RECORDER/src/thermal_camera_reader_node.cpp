#include <thermal_camera_reader/thermal_camera_reader.hpp>


int main(int argc, char** argv)
{
	ros::init(argc, argv, "thermal_camera_reader_node");
	ros::NodeHandle nh;

	Thermal_camera camera;

	ros::Rate r(camera.spin_rate); // 10 hz

	while(ros::ok()){
		if(camera.management())
			break;
			
		ros::spinOnce();
		r.sleep();
	}

	return 0;
}
