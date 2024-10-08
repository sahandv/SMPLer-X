# Visualisation code for SMPL-X model. This code is useful if you already have predictions.

import os
import sys
import os.path as osp
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
import torch
try:
    CUR_DIR = osp.dirname(os.path.abspath(__file__))
except NameError:
    CUR_DIR = os.getcwd()
sys.path.insert(0, osp.join(CUR_DIR, '..', 'main'))
sys.path.insert(0, osp.join(CUR_DIR , '..', 'common'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

JOINT_NAMES_DICT = {name: i for i, name in enumerate(JOINT_NAMES)}

# Load the SMPL-X model
model_path = 'common/utils/human_model_files'  # Update with the path to your SMPL-X models
model = smplx.create(model_path, model_type='smplx', gender='neutral', ext='npz')

# Load the parameters from the .npz file
data = np.load('/home/sahand/Downloads/smplx/00047_9.npz')

betas = torch.tensor(data['betas'], dtype=torch.float32)
body_pose = torch.tensor(data['body_pose'], dtype=torch.float32)
global_orient = torch.tensor(data['global_orient'], dtype=torch.float32)
transl = torch.tensor(data['transl'], dtype=torch.float32)
expression = torch.tensor(data['expression'], dtype=torch.float32)

# Add missing dimensions to the tensors
if betas.ndim == 1:
    betas = betas.unsqueeze(0)
if body_pose.ndim == 2:
    body_pose = body_pose.unsqueeze(0)
if global_orient.ndim == 1:
    global_orient = global_orient.unsqueeze(0)
if transl.ndim == 1:
    transl = transl.unsqueeze(0)
if expression.ndim == 1:
    expression = expression.unsqueeze(0)

# Reshape body_pose to include the batch dimension
body_pose = body_pose.view(1, -1, 3)

# Forward pass through the model
output = model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl, expression=expression)

# Extract joint positions
joints = output.joints.detach().cpu().numpy().squeeze()
print(joints.shape)
# Ankle joints (left and right)
left_knee = joints[4]  # Index for left ankle in SMPL-X
right_knee = joints[5]  # Index for right ankle in SMPL-X
left_ankle = joints[7]  # Index for left ankle in SMPL-X
right_ankle = joints[8]  # Index for right ankle in SMPL-X

bone_connections = [
    (JOINT_NAMES_DICT["pelvis"], JOINT_NAMES_DICT["spine1"]), (JOINT_NAMES_DICT["spine1"], JOINT_NAMES_DICT["spine2"]), (JOINT_NAMES_DICT["spine2"], JOINT_NAMES_DICT["spine3"]),  # Spine
    (JOINT_NAMES_DICT["pelvis"], JOINT_NAMES_DICT["left_hip"]), (JOINT_NAMES_DICT["left_hip"], JOINT_NAMES_DICT["left_knee"]), (JOINT_NAMES_DICT["left_knee"], JOINT_NAMES_DICT["left_ankle"]),  # Left leg
    (JOINT_NAMES_DICT["pelvis"], JOINT_NAMES_DICT["right_hip"]), (JOINT_NAMES_DICT["right_hip"], JOINT_NAMES_DICT["right_knee"]), (JOINT_NAMES_DICT["right_knee"], JOINT_NAMES_DICT["right_ankle"]),  # Right leg
    (JOINT_NAMES_DICT["left_ankle"], JOINT_NAMES_DICT["left_heel"]), 
    (JOINT_NAMES_DICT["right_ankle"], JOINT_NAMES_DICT["right_heel"]), 
    (JOINT_NAMES_DICT["left_ankle"], JOINT_NAMES_DICT["left_foot"]), 
    (JOINT_NAMES_DICT["left_foot"], JOINT_NAMES_DICT["left_big_toe"]), (JOINT_NAMES_DICT["left_foot"], JOINT_NAMES_DICT["left_small_toe"]),
    (JOINT_NAMES_DICT["right_ankle"], JOINT_NAMES_DICT["right_foot"]), 
    (JOINT_NAMES_DICT["right_foot"], JOINT_NAMES_DICT["right_big_toe"]), (JOINT_NAMES_DICT["right_foot"], JOINT_NAMES_DICT["right_small_toe"]),
    # Add more bones if necessary
]

# Visualize the 3D skeleton
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all joints
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', marker='o')
# Highlight ankle joints
ax.scatter([left_knee[0]], [left_knee[1]], [left_knee[2]], c='red', marker='x', s=100, label='Left Knee')
ax.scatter([right_knee[0]], [right_knee[1]], [right_knee[2]], c='green', marker='x', s=100, label='Right Knee')
ax.scatter([left_ankle[0]], [left_ankle[1]], [left_ankle[2]], c='red', marker='o', s=100, label='Left Ankle')
ax.scatter([right_ankle[0]], [right_ankle[1]], [right_ankle[2]], c='green', marker='o', s=100, label='Right Ankle')

# Draw bones
for bone in bone_connections:
    start, end = bone
    ax.plot([joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]], 'k-')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()