#！ /home/liao/anaconda3/envs/openmmlab/bin/python
from mmdet3d.apis import init_model, inference_detector
from sensor_msgs.msg import PointCloud2,Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import os
import rosbag
import rospy
import numpy as np
from open3d import geometry

def yaw_to_quaternion(yaw):
    roll=0
    pitch=0

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q = np.array([qx, qy, qz,qw])
    return q


def visualize_detection_result(detection_result,marker_publisher):

    # 创建一个Publisher，发布MarkerArray消息到'Rviz'话题

    marker_array = MarkerArray()

    for idx, detection in enumerate(detection_result):
        # 创建一个Marker
        marker = Marker()
        marker.header.frame_id = "lidar"  # 你需要将"base_link"替换为你所使用的坐标系
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.id = idx
        yaw=detection[6]
        quaternion=yaw_to_quaternion(yaw)
        marker.pose.position.x = detection[0]
        marker.pose.position.y = detection[1]
        marker.pose.position.z = detection[2]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = detection[4]
        marker.scale.y = detection[3]
        marker.scale.z = detection[5]
        marker.color.a = 0.7
        marker.color.r = 1.0  # 设置边界框的颜色
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker_array.markers.append(marker)

    # 发布PointCloud2消息和MarkerArray消息
    marker_publisher.publish(marker_array)


CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE)

config_file = os.path.join(ROOT,'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py')
checkpoint_file = os.path.join(ROOT,'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth')
model = init_model(config_file, checkpoint_file)

data_publisher = rospy.Publisher('/data', PointCloud2, queue_size=10)
image_publisher = rospy.Publisher('/image', Image, queue_size=10)
marker_publisher = rospy.Publisher('/detection_boxes', MarkerArray, queue_size=10)
if __name__ == '__main__':
    rospy.init_node('free_park_detection',anonymous=True)
    data_file = os.path.join(ROOT, 'data.bag')
    bag = rosbag.Bag(data_file)
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages():
        if topic=='/ouster_128':
            data_publisher.publish(msg)
            point_cloud_data = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"))
            pc_list=[]
            for x,y,z,i in point_cloud_data:
                one_pc=[x,y,z,i]
                pc_list.append(one_pc)
            data=np.asarray(pc_list)
            # print(data)
            result, data = inference_detector(model, data)
            pred_instances_3d=result.get('pred_instances_3d')
            bboxes_3d=pred_instances_3d.get('bboxes_3d').tensor.tolist()
            # print(bboxes_3d)
            visualize_detection_result(bboxes_3d,marker_publisher)

        elif topic=='/image':
            # print(type(msg))
            # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # ros_image = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            image_publisher.publish(msg)
        rospy.sleep(0.1)
    # print(result)