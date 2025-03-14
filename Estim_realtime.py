import csv
import os
from os.path import join, expanduser
import json
import numpy as np
from datetime import datetime
import time

from wedge_video import GelSightWedgeVideo, ORIGINAL_IMG_SIZE
from contact_force import ContactForce
from gripper_width import GripperWidth

from grasp_data import GraspData
from analytical_estimate import EstimateModulus, N_FRAMES
from train import ModulusModel
import queue

import matplotlib.pyplot as plt
plot_results = True
close_flag = False
save = True

# Load model
path_to_model = './model/test_run_1frame' # './model/rubber_only'
#path_to_model = './model/full_dataset'
with open(f'{path_to_model}/config.json', 'r') as file:
    config = json.load(file)
config['use_wandb'] = False

path_to_model_estim = './model/test_run_1frame_markers' # './model/rubber_only'
with open(f'{path_to_model}/config.json', 'r') as file:
    config = json.load(file)
config['use_wandb'] = False


# Prepare saving structure
home = expanduser("~")
if save==True:
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DIR_trial = f"{home}/Stiffness_estim/Data/{date_time}/"
    os.makedirs(DIR_trial, exist_ok=True) # Succeeds if folder already exists
    print(f"[INFO][SAVING] Made working directory: {DIR_trial}")

# 1. Record grasp
#       - We record grasps using src/collect_grasp_data.py

#wedge_video         =   GelSightWedgeVideo(IP="172.16.0.3", config_csv="./wedge_config/config_no_markers.csv") # Force-sensing finger
wedge_video_markers   =   GelSightWedgeVideo(IP="172.16.0.3", config_csv="./wedge_config/config_markers.csv") # Marker finger
contact_force       =   ContactForce(IP="172.16.0.1", port=8888)
gripper_width = GripperWidth()
#gripper_width._widths = np.linspace(0.02, 0.08, 243).tolist()
grasp_data          =   GraspData(wedge_video=wedge_video_markers, contact_force=contact_force, gripper_width=gripper_width)

grasp_data.start_stream(verbose=True, plot=True, plot_diff=True, plot_depth=False)

# 2. Compute analytical estimates
#       - Using grasp data, apply analytical algorithms to generate estimates
analytical_estimator    = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True)
modulus_model = ModulusModel(config,analytical_estimator=analytical_estimator)
modulus_model.load_model(path_to_model)
analytical_estimator.start_estimation(grasp_data,plot_results=True,verbose=True)
modulus_model.start_estimation(grasp_data, analytical_estimator, plot_results=True, verbose=True)

def end_program(event):
    print('Closing the app...')
    global close_flag
    close_flag = 1
    
## Plot some curves 
if plot_results:
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax1 = plt.subplots()
    fig.canvas.mpl_connect('close_event', end_program)
    ax2 = ax1.twinx()
    x_data, y1_data, y2_data, y3_data = [], [], [], []  # Data to be plotted
    line1, = ax1.plot([], [], 'b-', label="E_simple")  # Primary y-axis (left)
    line2, = ax1.plot([], [], 'r-', label="E_hertz")  # 
    line3, = ax2.plot([], [], 'k-', label="E_nn")  # Secondary y-axis (right)

    # Set up the axis limits and labels
    ax1.set_xlim(0, 100)  # You can adjust these limits based on your data
    ax1.set_xlabel("Time (s)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Young Modulus (MPa)", color="r")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("E NN (MPa)",  color="k")
    # Initialize the x value
    x = 0

if save == True:
    data_times, forcesvec, widthvec, contact_areas = [], [], [], []
    E_simple, E_hertz, E_hat = [], [], []
    start_time = datetime.now()
    while not close_flag:
        t0 = time.time()
        data_times.append((datetime.now() - start_time))
        forcesvec.append(np.squeeze(grasp_data.contact_force.forces(1)))
        widthvec.append(np.squeeze(grasp_data.gripper_width.widths(1)))
        contact_areas.append(analytical_estimator._contact_area_hertz)
        E_simple.append(analytical_estimator.E_simple) 
        E_hertz.append(analytical_estimator.E_hertz)
        E_hat.append(modulus_model.E_hat)

        # Try to run at 15Hz
        dt = (time.time() - t0)
        sleep_time = 1/15. - dt
        try:
            time.sleep(sleep_time)
            print(f"Learned estimate of Young's Modulus... {modulus_model.E_hat:.3e} Pa")
        except:
            print("lagged")

        if plot_results:
            # Pause to simulate real-time plotting            
            x_data.append(x)
            y1_data.append(analytical_estimator.E_simple*1e-6) 
            y2_data.append(analytical_estimator.E_hertz*1e-6)
            #y3_data.append(modulus_model.E_hat*5e-10) #

            # Update the line with the new data
            line1.set_xdata(x_data)
            line1.set_ydata(y1_data)
            line2.set_xdata(x_data)
            line2.set_ydata(y2_data)
            #line3.set_xdata(x_data)
            #line3.set_ydata(y3_data)
                    
            # Adjust the plot limits if needed
            if x >= ax1.get_xlim()[1]:  # Extend the x-axis limit
                ax1.set_xlim(0, x + 10)
            if max(y1_data) > ax1.get_ylim()[1] or max(y2_data) > ax1.get_ylim()[1]:  # Adjust y1-axis
                ax1.set_ylim(0, max(max(y1_data),max(y2_data)) + 0.5)
            #if max(y3_data) > ax2.get_ylim()[1]:  # Adjust y2-axis
            #    ax2.set_ylim(0, max(y3_data) + 0.5)
                    
            # Redraw the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
                        
            # Increment x for the next data point
            x += 1

    filename = join(DIR_trial, "data_all.csv")
    with open(filename, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["time","force","width","contact_area","E_simple","E_hertz","E_hat"])
        for i in range(len(data_times)): 
            writer.writerow([data_times[i], forcesvec[i], widthvec[i], contact_areas[i], E_simple[i], E_hertz[i], E_hat[i]])
    fd.close()
    grasp_data.save(DIR_trial)


#time.sleep(20)
grasp_data.end_stream(verbose=True)
analytical_estimator.stop_estimation(verbose=True)
modulus_model.stop_estimation(verbose=True)
