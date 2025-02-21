import os
import json
import numpy as np
import time

from wedge_video import GelSightWedgeVideo, ORIGINAL_IMG_SIZE
from contact_force import ContactForce
from gripper_width import GripperWidth

from grasp_data import GraspData
from analytical_estimate import EstimateModulus
from train import ModulusModel, N_FRAMES

# Load model
path_to_model = './model/test_run_1frame_estim' # './model/rubber_only'
with open(f'{path_to_model}/config.json', 'r') as file:
    config = json.load(file)
config['use_wandb'] = False
modulus_model = ModulusModel(config)
modulus_model.load_model(path_to_model)

# 1. Record grasp
#       - We record grasps using src/collect_grasp_data.py

wedge_video         =   GelSightWedgeVideo(IP="172.16.0.3", config_csv="./wedge_config/config_no_markers.csv") # Force-sensing finger
#wedge_video_markers   =   GelSightWedgeVideo(IP="172.16.0.3", config_csv="./wedge_config/config_markers.csv") # Marker finger
contact_force       =   ContactForce(IP="172.16.0.1", port=8888)
gripper_width = GripperWidth()
gripper_width._widths = np.linspace(0.02, 0.08, 243).tolist()
grasp_data          =   GraspData(wedge_video=wedge_video, contact_force=contact_force, gripper_width=gripper_width)

grasp_data.start_stream(verbose=True, plot=True, plot_diff=False, plot_depth=False)

# 2. Compute analytical estimates
#       - Using grasp data, apply analytical algorithms to generate estimates
analytical_estimator    = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True)
analytical_estimator.start_estimation(grasp_data,plot_results=True,verbose=True)
modulus_model.start_estimation(grasp_data, E_hat_simple=analytical_estimator.E_simple, E_hat_hertz=analytical_estimator.E_hertz, plot_results=True, verbose=True)

time.sleep(10)
grasp_data.end_stream(verbose=True)
analytical_estimator.stop_estimation(verbose=True)