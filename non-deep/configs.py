import argparse
import numpy as np
import os,sys

PROJECT_ROOT='/local/MI/zqj/video_highlights/'
PROJECT_METAROOT=os.path.join(PROJECT_ROOT,'metadata')
LOG_ROOT=os.path.join(PROJECT_ROOT,'log')
BAIDU_VH_ROOT='/local/MI/temporal_action_localization/data/baidu_VH/'
MIN_FEATURE=-3
METRICS='average_precision'
ALTER=[24]
grid_search_params={'xgb':{
    'max_depth':[3,6,9],
    'reg_alpha':[5,10,15],
    'reg_lambda':[5,10,15],
    'gamma':[1,4,7],
    'subsample':[0.5,0.8],},
                   'svm':{
    'C':[1,5,20],
    'gamma':[0.1,1,10]},
    }

def get_args_watershed():
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--process', type=str, nargs='+',default='train')
    parser.add_argument('--dataset', type=str, default='baidu_VH',choices=['baidu_VH','summe'])
    parser.add_argument('--thread', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='xgb')
    parser.add_argument('--modality',type=str,default='image')
    # Model
    parser.add_argument('--if_grid_search', type=bool, default=False)
    parser.add_argument('--model_kernel',type=str,default='rbf')
    parser.add_argument('--grid_search_sample_rate', type=float, default=0.1)
    parser.add_argument('--create_curves', type=bool, default=True)
    parser.add_argument('--save_curves', type=bool, default=True)
    parser.add_argument('--clips',type=float,nargs='+',default=[0.5,0.2])
    parser.add_argument('--type',type=str,default='smooth')
    parser.add_argument('--save_path',type=str,default='tmp_results/final_{}.json')
    parser.add_argument('--alter',type=int,nargs='+',default=[16])
    parser.add_argument('--nms',type=float,default=0.5)
    parser.add_argument('-v',action='store_true')
    # Train
    parser.add_argument('--get_final_test', type=bool, default=False)

    # load epoch
    parser.add_argument('--epoch', type=int, default=2)

    args = parser.parse_args()
    return args

def get_args(mode='watershed'):
    if mode=='watershed':
        return get_args_watershed()