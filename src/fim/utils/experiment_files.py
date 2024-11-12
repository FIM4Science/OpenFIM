import os
import re
import time
import json
import torch
import shutil
import subprocess
from pathlib import Path
from fim import results_path
from typing import Union,Tuple,List
from dataclasses import dataclass, asdict

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
#    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

class ExperimentsFiles:
    """
    Defines experiment folders with sub dirs for 
        -tensorboard dir
        -checkpoint storage
        -sample folders
    Defines paths str for
        - parametes yamls
        - plot path
    """
    def __init__(self,experiment_dir=None,experiment_indentifier=None,delete=False):
        self.delete = delete
        self.define_experiment_folder(experiment_dir,experiment_indentifier)
        self._create_directories()

    def define_experiment_folder(self,experiment_dir=None,experiment_indentifier=None):
        if experiment_dir is None:
            from fim import results_path
            results_dir = str(results_path)
            self.experiment_indentifier = experiment_indentifier
            if self.experiment_indentifier is None:
                self.experiment_indentifier = str(int(time.time()))
            self.experiment_dir = os.path.join(results_dir, self.experiment_indentifier)
        else:
            self.experiment_dir = experiment_dir

        self.tensorboard_dir = os.path.join(self.experiment_dir, "logs")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.sample_dir = os.path.join(self.experiment_dir, "sample")
        self.params_yaml = os.path.join(self.experiment_dir, "params.yaml")  
        self.plots_path  = os.path.join(self.experiment_dir, "plots_{0}.png")
        self.metrics_path = os.path.join(self.experiment_dir, "plots_{0}.json")

    def _create_directories(self):
        if Path(self.experiment_dir).exists():
            if self.delete:
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)
                os.makedirs(self.tensorboard_dir)
                os.makedirs(self.checkpoints_dir)
                os.makedirs(self.sample_dir)
        else:
            print("Creating Experiment Directory")
            os.makedirs(self.experiment_dir)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.sample_dir)

    def get_lightning_checkpoint_path(self,checkpoint_type:str="best"):
        """
        Checks for lightning checkpoints in experiment folders and returns 
        checkpoint type
        """
        checkpoints_path = Path(self.checkpoints_dir)
        all_checkpoints = os.listdir(checkpoints_path)

        if len(all_checkpoints) == 0:
            raise Exception("No Model Found!")
        
        for checkpoint_name in all_checkpoints:
            checkpoint_path = checkpoints_path / checkpoint_name
            if checkpoint_type in checkpoint_name:
                return checkpoint_path
 
        print("CHECKPOINT TYPE {0} NOT FOUND, RETURNING".format(checkpoint_type))
        print(all_checkpoints[0])
        return checkpoint_path / all_checkpoints[0]