#！ /home/liao/anaconda3/envs/openmmlab/bin/python
from mmdet3d.apis import init_model, inference_detector
from sensor_msgs.msg import PointCloud2,Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import os
import rosbag
import rospy
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
# from dbscan import get_vertex,dbscan
# from sklearn.cluster import HDBSCAN
# from cuml import DBSCAN
# import pandas as pd
# import cudf
def rm_ground_roof_points(point_cloud, h_ground, h_roof):
    points = point_cloud
    # remove ground and roof points
    points_ground = points[points[:, 2] < h_ground]
    points_roof = points[points[:, 2] > h_roof]

    points = points[points[:, 2] >= h_ground]
    points = points[points[:, 2] <= h_roof]

    return points, points_ground, points_roof

def dbscan(points):
    # 对点云进行标准化处理
    # points = StandardScaler().fit_transform(points)
    # 创建DBSCAN对象并进行聚类
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    dbscan.fit_predict(points)
    # 获取每个点的簇标签（-1表示噪声点）
    labels = dbscan.labels_
    return labels

def get_vertex(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    top_left_front = min_coords
    top_right_front = [max_coords[0], min_coords[1], max_coords[2]]
    bottom_left_front = [min_coords[0], max_coords[1], max_coords[2]]
    bottom_right_front = [max_coords[0], max_coords[1], min_coords[2]]
    top_left_back = [min_coords[0], min_coords[1], max_coords[2]]
    top_right_back = max_coords
    bottom_left_back = [min_coords[0], max_coords[1], min_coords[2]]
    bottom_right_back = [max_coords[0], min_coords[1], min_coords[2]]
    box = [bottom_right_back, bottom_right_front, bottom_left_front, bottom_left_back,
           top_right_back, top_right_front, top_left_front, top_left_back]
    l = max_coords[0] - min_coords[0]
    w = max_coords[1] - min_coords[1]
    h = max_coords[2] - min_coords[2]
    h_max, h_min = max_coords[2], min_coords[2]
    center = (max_coords + min_coords) / 2

    return box, l, w, h, h_max, h_min, center

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

# if __name__ == '__main__':
#     rospy.init_node('free_park_detection',anonymous=True)
#     data_file = os.path.join(ROOT, 'data.bag')
#     bag = rosbag.Bag(data_file)
#     bridge = CvBridge()
    # for topic, msg, t in bag.read_messages():
    #     if topic=='/image':
    #         cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         timestamp = str(msg.header.stamp.to_sec())
    #         image_filename = os.path.join(ROOT,'image', f'{timestamp}.jpg')
    #         # print(type(msg))
    #         # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #         # ros_image = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    #         cv2.imwrite(image_filename, cv_image)
    # print(result)
if __name__ == '__main__':
    rospy.init_node('free_park_detection',anonymous=True)
    data_file = os.path.join(ROOT, 'data.bag')
    bag = rosbag.Bag(data_file)
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages():
        print(topic)
        # detection = []
        # if topic=='/ouster_128':
        #
        #     data_publisher.publish(msg)
        #     point_cloud_data = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"))
        #     pc_list=[]
        #     for x,y,z,i in point_cloud_data:
        #         one_pc=[x,y,z,i]
        #         pc_list.append(one_pc)
        #     data=np.asarray(pc_list)
        #     point=data[:,:3]
        #     points, points_ground, point_roof = rm_ground_roof_points(point, 0.3, 1.8)
        #     points=points[(points[:,0]<=10) & (points[:,1]<=10)]
        #     # points_df=pd.DataFrame(points)
        #     # points_gpu=cudf.DataFrame.from_pandas(points_df)
        #     labels = dbscan(points)
        #     print('labels',labels)
        #     n_clusters = len(set(labels)) - (1 if -1 in labels else 0) + 1
        #     cls_num = np.arange(-1, n_clusters - 1)
        #     # print(len(cls_num))
        #     for i in cls_num:
        #         cluster_points = points[labels == i]
        #         if i == -1 or len(cluster_points) < 5:
        #             continue
        #         _, l, w, h, h_max, h_min, center = get_vertex(cluster_points)
        #         box=[center[0],center[1],center[2],l,w,h,0]
        #         detection.append(box)
        #     print(len(detection),timestamp)
        #     detection_3d=np.array(detection)
        #     # result, data = inference_detector(model, data)
        #     # pred_instances_3d=result.get('pred_instances_3d')
        #     # bboxes_3d=np.array(pred_instances_3d.get('bboxes_3d').tensor.cpu())
        #     visualize_detection_result(detection_3d, marker_publisher)
            # print('bboxes_3d',bboxes_3d)
            # if len(bboxes_3d)!=0:
            #     bboxes_3d_center=bboxes_3d[:,:3]
            #     cluster_labels = dbscan.fit_predict(bboxes_3d_center)
            #     print('cluster_labels',cluster_labels)
            #     merged_boxes = []
            #     for label in np.unique(cluster_labels):
            #         if label == -1:
            #             continue
            #         cluster_boxes = bboxes_3d[cluster_labels == label]
            #         print('cluster_bboxes',cluster_boxes)
            #     merged_box = [
            #         np.min(cluster_boxes[:, 0]),
            #         np.min(cluster_boxes[:, 1]),
            #         np.max(cluster_boxes[:, 2]),
            #         np.max(cluster_boxes[:, 3])
            #     ]
            #
            #     merged_boxes.append(merged_box)
            # print(bboxes_3d)


        # if topic=='/image':
        #     # print(type(msg))
        #     timestamp_img = str("%.9f" % msg.header.stamp.to_sec())

            # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # # ros_image = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            # file=os.path.join(ROOT,'image_origin',timestamp+'.jpg')
            # cv2.imwrite(file,cv_image)
        #     image_publisher.publish(msg)
        # rospy.sleep(0.1)
