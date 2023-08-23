import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d
import argparse


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        rgb = np.random.rand(3)  # 随机生成RGB值
        colors.append(rgb)
    return np.array(colors)


def dbscan(points):
    # 对点云进行标准化处理
    # points = StandardScaler().fit_transform(points)
    # 创建DBSCAN对象并进行聚类
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    dbscan.fit(points)
    # 获取每个点的簇标签（-1表示噪声点）
    labels = dbscan.labels_
    return labels


def rm_ground_roof_points(point_cloud, h_ground, h_roof):
    points = np.asarray(point_cloud.points)
    # remove ground and roof points
    points_ground = points[points[:, 2] < h_ground]
    points_roof = points[points[:, 2] > h_roof]

    points = points[points[:, 2] >= h_ground]
    points = points[points[:, 2] <= h_roof]

    return points, points_ground, points_roof


def vis_points(points, window_name, points_size, background_color, pcd=None):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1500, height=800)
    if pcd == None:
        pcd = open3d.open3d.geometry.PointCloud()
        pcd.points = open3d.open3d.utility.Vector3dVector(points)
    opt = vis.get_render_option()
    opt.point_size = points_size
    opt.background_color = np.asarray(background_color)
    vis.add_geometry(pcd)
    vis.run()
    vis.clear_geometries()
    vis.destroy_window()
    return pcd


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


def add_park_set(center, park_size):
    '''画车位'''
    car_vertex1 = [center[0] - park_size[1] / 2, center[1] - park_size[0] / 2, 0.2]
    car_vertex2 = [center[0] + park_size[1] / 2, center[1] + park_size[0] / 2, 0.2]
    car_vertex3 = [center[0] + park_size[1] / 2, center[1] - park_size[0] / 2, 0.2]
    car_vertex4 = [center[0] - park_size[1] / 2, center[1] + park_size[0] / 2, 0.2]
    park_vertex = [car_vertex1, car_vertex2, car_vertex3, car_vertex4]
    park_box = np.array(
        [[0, 2], [1, 3], [0, 3], [1, 2]])

    colors = np.array([[1, 1, 1] for j in range(len(park_box))])  # 车位颜色
    park_set = open3d.geometry.LineSet()
    park_set.lines = open3d.utility.Vector2iVector(park_box)

    park_set.colors = open3d.utility.Vector3dVector(colors)
    park_set.points = open3d.utility.Vector3dVector(park_vertex)
    vis.add_geometry(park_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--h_ground', type=float, default=0.3)
    parser.add_argument('--h_roof', type=float, default=1.8)
    parser.add_argument('--pcd_path', type=str, default='../pcd/1684331296100.pcd')
    parser.add_argument('--point_size', type=int, default=3)
    parser.add_argument('--park_size', type=list, default=[5, 2.5])

    args = parser.parse_args()
    h_ground = args.h_ground
    h_roof = args.h_roof
    pcd_path = args.pcd_path
    point_size = args.point_size
    park_size = args.park_size

    point_cloud = open3d.io.read_point_cloud(pcd_path)
    vis_points(point_cloud.points, window_name='原图', points_size=1, background_color=[0, 0, 0])
    points, points_ground, point_roof = rm_ground_roof_points(point_cloud, h_ground, h_roof)
    labels = dbscan(points)
    # pcd = vis_points(points, window_name='滤除后可视化', points_size=1, background_color=[0, 0, 0])
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) + 1  # 含噪声点
    colors = generate_colors(n_clusters)
    colored_points = colors[labels % len(colors)]
    # pcd.colors = open3d.utility.Vector3dVector(colored_points)
    # pcd = vis_points(points, window_name='聚类后可视化', points_size=5, background_color=[255, 255, 255], pcd=pcd)
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name="box", width=1500, height=800)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray(np.array([0, 0, 0]))
    # vis.add_geometry(pcd)
    coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    cls_num = np.arange(-1, n_clusters - 1)

    print(len(cls_num))
    print(cls_num)
    pcd_ground = open3d.open3d.geometry.PointCloud()
    pcd_ground.points = open3d.open3d.utility.Vector3dVector(points_ground)
    vis.add_geometry(pcd_ground)
    for i in cls_num:
        cluster_points = points[labels == i]
        if i == -1 or len(cluster_points) < 5:
            continue
        pcd_cluster = open3d.open3d.geometry.PointCloud()
        pcd_cluster.points = open3d.open3d.utility.Vector3dVector(cluster_points)
        box, l, w, h, h_max, h_min, center = get_vertex(cluster_points)
        if h_max > 1.5 and h > 0.5 and l < 1.8 and w < 1.8:  # 柱子
            pcd_cluster.paint_uniform_color([0, 1, 0])
        if l > 4 or w > 4:  # 墙
            pcd_cluster.paint_uniform_color([1, 1, 0])
        elif l > 1.0 and w > 1.0:  # 车
            add_park_set(center, park_size)
            pcd_cluster.paint_uniform_color([0.7, 0, 0.8])
        # 画框
        lines_box = np.array(
            [[0, 1], [1, 3], [2, 3], [4, 1], [2, 4], [4, 5],
             [5, 0], [5, 7], [6, 7], [0, 6], [7, 2], [3, 6]])
        colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
        line_set = open3d.geometry.LineSet()
        line_set.lines = open3d.utility.Vector2iVector(lines_box)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        line_set.points = open3d.utility.Vector3dVector(box)
        vis.add_geometry(pcd_cluster)
        vis.add_geometry(line_set)
    vis.run()
    vis.clear_geometries()
    vis.destroy_window()
