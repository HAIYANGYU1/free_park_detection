The project works as a ROS node. So please install ROS-noetic first.
http://wiki.ros.org/noetic/Installation/Ubuntu#Installation



Install MMdetection3D:  
https://mmdetection3d.readthedocs.io/en/latest/get_started.html  
Command:  
conda create --name openmmlab python=3.8 -y  
conda activate openmmlab  
conda install pytorch torchvision -c pytorch(On GPU platform)  
conda install pytorch torchvision cpuonly -c pytorch(On CPU platform)  
pip install -U openmim  
mim install mmengine  
mim install 'mmcv>=2.0.0rc4'  
mim install 'mmdet>=3.0.0'  
mim install "mmdet3d>=1.1.0rc0"  
(download the config and checkpoint files)  
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .  


