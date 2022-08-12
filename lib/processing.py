import sys
sys.path.append('/home/xuchengjun/ZXin/EARN')
import scipy.io as io
from IPython import embed
from path import Path
from lib.normlize import pose_normalization, change_pose, vector_pose_normalization
from utils import augment_pose, show_3d_results
import csv 
import os
import numpy as np
import pandas as pd

"""
UTD-MHAD(single view)
最大的帧数为125,最小的帧数为41,27动作
features_structure['d_skel'] --> 取出3D数据
action_label = str(file).split('/')[-1].split('a')[1].split('_')[0]

UTD-MHAD(multi-view)
最大的帧数为,最小的帧数为,6动作 --> [catch (0), draw circle (1), draw tick (2), draw triangle (3), konck (4), throw (5)]
features_structure['S_K2'][0]['world'][0]  --> 取出3D数据
"""

#UTD-MAHD
# ----------------------------------------------------------------------------------
# 20
# headers = ["head_x","head_y","head_z",
#             'neck_x','neck_y','neck_z',
#             'spine_x','spine_y','spine_z',
#             'hip_center_x','hip_center_y','hip_center_z',
#             'left_shoulder_x','left_shoulder_y','left_shoulder_z',
#             'left_elbow_x','left_elbow_y','left_elbow_z',
#             'left_wrist_x','left_wrist_y','left_wrist_z',
#             'left_hand_x','left_hand_y','left_hand_z',
#             'right_shoulder_x','right_shoulder_y','right_shoulder_z',
#             'right_elbow_x','right_elbow_y','right_elbow_z',
#             'right_wrist_x','right_wrist_y','right_wrist_z',
#             'right_hand_x','right_hand_y','right_hand_z',
#             'left_hip_x','left_hip_y','left_hip_z',
#             'left_knee_x','left_knee_y','left_knee_z',
#             'left_ankle_x','left_ankle_y','left_ankle_z',
#             'left_foot_x','left_foot_y','left_foot_z',
#             'right_hip_x','right_hip_y','right_hip_z',
#             'right_knee_x','right_knee_y','right_knee_z',
#             'right_ankle_x','right_ankle_y','right_ankle_z',
#             'right_foot_x','right_foot_y','right_foot_z',
#           ]

#25
# headers = ['root_x','root_y','root_z',
#            'spine_mid_x','spine_mid_y','spine_mid_z',
#            'neck_x','neck_y','neck_z',
#            "head_x","head_y","head_z",
#            'right_shoulder_x','right_shoulder_y','right_shoulder_z',
#            'right_elbow_x','right_elbow_y','right_elbow_z',
#            'right_wrist_x','right_wrist_y','right_wrist_z',
#            'right_hand_x','right_hand_y','right_hand_z',
#            'left_shoulder_x','left_shoulder_y','left_shoulder_z',
#            'left_elbow_x','left_elbow_y','left_elbow_z',
#            'left_wrist_x','left_wrist_y','left_wrist_z',
#            'left_hand_x','left_hand_y','left_hand_z',                      
#            'right_hip_x','right_hip_y','right_hip_z',
#            'right_knee_x','right_knee_y','right_knee_z',
#            'right_ankle_x','right_ankle_y','right_ankle_z',
#            'right_foot_x','right_foot_y','right_foot_z',
#            'left_hip_x','left_hip_y','left_hip_z',
#            'left_knee_x','left_knee_y','left_knee_z',
#            'left_ankle_x','left_ankle_y','left_ankle_z',
#            'left_foot_x','left_foot_y','left_foot_z',
#            'spine_shoulder_x','spine_shoulder_y','spine_shoulder_z', 
#            'right_thumb_x','right_thumb_y','right_thumb_z',  
#            'right_hand_tip_x','right_hand_tip_y','right_hand_tip_z', 
#            'left_thumb_x','left_thumb_y','left_thumb_z',   
#            'left_hand_tip_x','left_hand_tip_y','left_hand_tip_z',                           
# ]


# # /media/xuchengjun/datasets/UTD-MAD/cs2/test
# # /media/xuchengjun/datasets/UTD-MHAD-MV/sub1/front/catch
# # catch (0), draw_circle (1), draw_tick (2), draw_triangle (3), knock (4), throw (5)
# # front, left_45_degree, left_90_degree, right_45_degree, right_90_degree
# root_path = '/media/xuchengjun/datasets/UTD-MHAD-MV/sub1/front/catch'
# files = Path(root_path).files()
# csv_dir = '/media/xuchengjun/datasets/UTD-MHAD-MV/new_data/aug1'

# max_frame = 0
# min_frame = 300
# frame_list = []
# # angle_list = [5 * i for i in range(1,73)]
# angle_list = [0]

# sub_label = root_path.split('/')[-3].split('sub')[1]

# action_label = 5

# #multi-view
# for angle in angle_list:
#     for idx, file in enumerate(files):
#         features_structure = io.loadmat(file)
#         frames = features_structure['S_K2'][0]['world'][0].shape[2]   #取出数据，取出帧数
#         angle_label = angle
#         # if frames > max_frame:
#         #     max_frame = frames
#         # if frames < min_frame:
#         #     min_frame = frames
#         # frame_list.append(frames)
#         # print(idx)

#         # new_pose --> (joint_num,3)
#         new_pose = (features_structure['S_K2'][0]['world'][0] * 100).transpose(2,1,0).transpose(0,2,1)

#         pose_frame_list = []
#         for i in range(frames):
#             aug_pose = augment_pose(pred_3d_bodys=new_pose[i], angles=0, root_idx=0)
#             normlize_pose = vector_pose_normalization(aug_pose[:,:3], 0)
#             show_3d_results(pred_3d_poses=normlize_pose)
#             # normlize_pose = pose_normalization(aug_pose[:,:3])
#             # changed_pose = change_pose(normlize_pose)  # 拉成一维
#             # pose_frame_list.append(changed_pose)

        
#         # csv_file = os.path.join(csv_dir, str(action_label) + '_aug' + str(angle_label) + '_s' + str(sub_label) + '_' + str(idx) + '.csv')
#         # with open(csv_file, 'w', newline='') as csvfile:
#         #     csv_writer = csv.writer(csvfile)
#         #     csv_writer.writerow(headers)
#         #     csv_writer.writerows(pose_frame_list)

# print('max_frame, min_frame:', max_frame, min_frame)




# single view
# for idx, file in enumerate(files):
#     features_structure = io.loadmat(file)
#     frames = features_structure['d_skel'].shape[2]   #取出数据，取出帧数
#     action_label = str(file).split('/')[-1].split('a')[1].split('_')[0]  #
#     if frames > max_frame:
#         max_frame = frames
#     if frames < min_frame:
#         min_frame = frames
#     frame_list.append(frames)
#     print(idx)

#     # new_pose --> (joint_num,3)
#     # new_pose = (features_structure['d_skel'] * 100).transpose(2,1,0).transpose(0,2,1)
#     # pose_frame_list = []
#     # for i in range(frames):
#     #     normlize_pose = pose_normalization(new_pose[i])
#     #     changed_pose = change_pose(normlize_pose)
#     #     pose_frame_list.append(changed_pose)
    
#     # csv_file = os.path.join(csv_dir, action_label + '_' + str(idx) + '.csv')
#     # with open(csv_file, 'w', newline='') as csvfile:
#     #     csv_writer = csv.writer(csvfile)
#     #     csv_writer.writerow(headers)
#     #     csv_writer.writerows(pose_frame_list)
#     # print(f"current --> {idx}")
# -------------------------------------------------------------------------------




# MSR 3d action
headers = ['left_shoulder_x','left_shoulder_y','left_shoulder_z',
            'right_shoulder_x','right_shoulder_y','right_shoulder_z',
            'neck_x','neck_y','neck_z',
            'spine_x','spine_y','spine_z',
            'left_hip_x','left_hip_y','left_hip_z',
            'right_hip_x','right_hip_y','right_hip_z',
            'hip_center_x','hip_center_y','hip_center_z',
            'left_elbow_x','left_elbow_y','left_elbow_z',
            'right_elbow_x','right_elbow_y','right_elbow_z',
            'left_wrist_x','left_wrist_y','left_wrist_z',
            'right_wrist_x','right_wrist_y','right_wrist_z',
            'left_hand_x','left_hand_y','left_hand_z',
            'right_hand_x','right_hand_y','right_hand_z',
            'left_knee_x','left_knee_y','left_knee_z',
            'right_knee_x','right_knee_y','right_knee_z',
            'left_ankle_x','left_ankle_y','left_ankle_z',
            'right_ankle_x','right_ankle_y','right_ankle_z',
            'left_foot_x','left_foot_y','left_foot_z',
            'right_foot_x','right_foot_y','right_foot_z',
            "head_x","head_y","head_z",
]

root_path = '/media/xuchengjun/datasets/MSRAction3D/cs2/all_data'
files = Path(root_path).files()
csv_dir = '/media/xuchengjun/datasets/MSRAction3D/cs2/train_csv'

train_sub = ['s01','s03','s05','s07','s09']
test_sub = ['s02','s04','s06','s08','s10']

idx = 0
for i,file_path in enumerate(files):

    txt_data = pd.read_csv(file_path,sep='\t',header=None)
    txt_data.columns=['x']

    xx = []
    yy = []
    zz = []
    pose_frame_list = []

    csv_name = str(file_path).split('/')[-1].split('.')[0]
    action_label = int(csv_name.split('_')[0].split('a')[1])

    action_idx = csv_name.split('_')[2]
    subject = csv_name.split('_')[1]
    if subject in train_sub:
        for j in range(1,1+len(txt_data)):

            coord = txt_data['x'][j-1].split()
            print(csv_name)
            xx.append(float(coord[0]))
            yy.append(float(coord[1]))
            zz.append(float(coord[2]))

            if j % 20 == 0:

                total = []
                for k in range(20):
                    total.append([xx[k],yy[k],zz[k]])
                
                total = np.array(total)
                normlize_pose = pose_normalization(total)
                if normlize_pose is None:
                    break
                changed_pose = change_pose(normlize_pose)
                pose_frame_list.append(changed_pose)
                xx = []
                yy = []
                zz = [] 

    if len(pose_frame_list) < 10:
        continue
    csv_file = os.path.join(csv_dir,str(action_label)+'_'+str(subject)+ '_' + action_idx + '.csv')
    with open(csv_file,'w',newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        csv_writer.writerows(pose_frame_list)
    idx+=1

    # print(f'第{i}个 。。')
 




