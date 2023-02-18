import subprocess
import time
import os
import shutil
import pickle
import glob

import numpy as np

def eval_heuristic(res):
    success = res['success']
    time_taken = res['total_sim_time_taken']
    collisions = res['total_collisions']
    robot_motion = res['robot_motion_energy']
    ped_dist = np.exp(-res['closest_pedestrian_distance']*2).sum()
    costs_vec = [5.1-5*success,time_taken,collisions+1,robot_motion,ped_dist]
    return np.prod(costs_vec)**(1/len(costs_vec)), costs_vec

def exec_seqs(params,base_name='local',set_s=[],log_file=[]):
    base_path = 'tests/socnav/{}_social_force'.format(base_name)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    os.environ['PYTHONPATH'] = '.'

    test_s = subprocess.Popen(['python', 'tests/test_episodes.py','--dir',base_name] + set_s,env=os.environ)#,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    time.sleep(0.5)
    joystick_s = subprocess.Popen(['python', 'joystick/joystick_client.py','--algo','socialforce','--dir',base_name],env=os.environ)#,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    time.sleep(0.5)
    sf_exec = subprocess.Popen(['joystick/social_force/social_force'] + ['{:.3f}'.format(_) for _ in list(np.exp(params[:-1])) + [params[-1]]])#,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    poll = test_s.poll()
    while poll is None:
        poll = test_s.poll()
        time.sleep(0.1)
    poll = joystick_s.poll()
    while poll is None:
        poll = joystick_s.poll()
        time.sleep(0.1)
    test_s.kill()
    joystick_s.kill()
    sf_exec.kill()

    total_costs_vec = []
    for out_f in sorted(glob.glob(os.path.join(base_path,'*','*.pkl'))):
        with open(out_f,'rb') as fp:
            res = pickle.load(fp)
            
        folder = out_f.split('/')[-2]
        res['filename'] = folder
        res['config_used'] = params

        cost, costs_vec = eval_heuristic(res)
        total_costs_vec.append(costs_vec)
        log_file.append(res)
        
    total_costs = np.prod(total_costs_vec,axis=1)**(1/len(costs_vec))
    total_costs = total_costs.mean()

    return total_costs