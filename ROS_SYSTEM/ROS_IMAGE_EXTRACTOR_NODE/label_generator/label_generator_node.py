from cv_bridge import CvBridge
import cv2
import numpy as np
import csv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class LabelGenerator(Node):

    def __init__(self):
        super().__init__('label_generator')
        self.subscription = self.create_subscription(Image,'/republished/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image,'/republished/depth/image_raw', self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo,'/d400/color/camera_info', self.cam_info_callback, 10)
        self.depth_caminfo_sub = self.create_subscription(CameraInfo,'/d400/aligned_depth_to_color/camera_info', self.depth_cam_info_callback, 10)
        self.thermal_sub = self.create_subscription(Image,'/thermal_img', self.thermal_callback, 10)
        self.thermal_norm_sub = self.create_subscription(Image,'/thermal_img_normalized', self.thermal_norm_callback, 10)
        self.subscription
        self.depth_sub
        self.caminfo_sub
        self.thermal_sub
        self.thermal_norm_sub

        self.declare_parameter('destination', '/home/simone/Documenti/labelstudio/images/')
        self.address = self.get_parameter('destination').get_parameter_value().string_value

        self.declare_parameter('fps_count', 15)
        self.fps_saved = self.get_parameter('fps_count').get_parameter_value().integer_value

        self.br = CvBridge()
        self.fps_count = 0
        self.im_count = 0
        self.start_color = 1
        self.start_depth = 1

        self.depth_im = np.zeros((1080, 1920))
        self.thermal_im = np.zeros((240, 320))
        self.therma_norm_im = np.zeros((240, 320))


    def cam_info_callback(self, msg):
        fx = msg.p[0]
        cx = msg.p[2]
        fy = msg.p[5]
        cy = msg.p[6]

        intrinsic_par = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

        if self.start_color == 1:
            f = open(self.address + "color_caminfo.csv", 'w')
            writer = csv.writer(f)
            writer.writerow(intrinsic_par[0, :])
            writer.writerow(intrinsic_par[1, :])
            writer.writerow(intrinsic_par[2, :])
            f.close()
            self.start_color = 0
   
    def depth_cam_info_callback(self, msg):
        fx = msg.p[0]
        cx = msg.p[2]
        fy = msg.p[5]
        cy = msg.p[6]

        intrinsic_par = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

        if self.start_depth == 1:
            f = open(self.address + "depth_caminfo.csv", 'w')
            writer = csv.writer(f)
            writer.writerow(intrinsic_par[0, :])
            writer.writerow(intrinsic_par[1, :])
            writer.writerow(intrinsic_par[2, :])
            f.close()
            self.start_depth = 0

    def depth_callback(self, msg):
        self.depth_im = self.br.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def thermal_callback(self, msg):
        self.thermal_im = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def thermal_norm_callback(self, msg):
        self.thermal_norm_im = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        self.get_logger().info('Receiving video frame')
        self.fps_count = self.fps_count + 1

        if self.fps_count > self.fps_saved:
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite(self.address + "img_" + str(self.im_count) + ".png", current_frame)
            cv2.imwrite(self.address + "depth_" + str(self.im_count) + ".png", self.depth_im)
            cv2.imwrite(self.address + "thermal_" + str(self.im_count) + ".png", self.thermal_im)
            cv2.imwrite(self.address + "thermal_norm_" + str(self.im_count) + ".png", self.thermal_norm_im)
            self.im_count = self.im_count + 1
            self.fps_count = 0
        

def main():
    rclpy.init()

    lgenerator = LabelGenerator()

    rclpy.spin(lgenerator)

    lgenerator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
