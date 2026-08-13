import os
import cmaes
import scipy
import numpy as np
import ipynbname
import optuna
import optunahub
from optuna.samplers import RandomSampler
from optuna.exceptions import TrialPruned

class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float('inf')
        self.trials_without_improvement = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # USA trial.values (lista com [wape_RUL, wape_HI])
        # Somamos os valores para avaliar a melhoria geral
        current_score = sum(trial.values)

        if current_score < self.best_score:
            self.best_score = current_score
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.patience:
            print(f"O estudo parou! O erro não diminui há {self.patience} iterações.")
            study.stop()

def SelSampler(mode=None,n_startup_trials=1000, seed=None):
    '''
    modes available: auto, GP, NSGAII, random, tpe

    None: sampler default, suport multivariate optimization
    GP: suport multivariate optimization
    Auto: suport multivariate optimization
    NSGAII: poorly suport multivariate optimization
    Random: poorly suport multivariate optimization
    TPE: suport multivariate optimization

    '''
    module = optunahub.load_module(package="samplers/auto_sampler")
    if mode is None or mode.lower() == 'none':
        sampler = None
    elif mode.lower() == 'gp':
        sampler = optuna.samplers.GPSampler(seed=seed)
    elif mode.lower() == 'auto':
        sampler = module.AutoSampler(seed=seed)    
    elif mode.lower() == 'nsgaii':
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    elif mode.lower() == 'tpe':
        sampler = optuna.samplers.TPESampler(multivariate=True,group=True,n_startup_trials=n_startup_trials,seed=seed)
    elif mode.lower() == 'random':
        sampler=RandomSampler(seed=seed)
    return sampler

def df_ParamsTable(names,FileName):
    if len(names) == 13 or len(names) == 11:
        study_dir = f'Optimization/{FileName}/multi/'
        out_path = f'Optimization/{FileName}/multi/Optimization.csv'
        study_dir = os.path.normpath(study_dir)
        out_path = os.path.normpath(out_path)
        os.makedirs(study_dir, exist_ok=True)
    elif len(names) == 12 or len(names) == 10:
        study_dir = f'Optimization/{FileName}/mono/'
        out_path = f'Optimization/{FileName}/mono/Optimization.csv'
        study_dir = os.path.normpath(study_dir)
        out_path = os.path.normpath(out_path)
        os.makedirs(study_dir, exist_ok=True)

    return study_dir, out_path