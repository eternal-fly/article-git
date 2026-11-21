import os
from experiment_common import collect_jobs, train_and_analyze, shot_noise_bootstrap
collect_jobs(os.environ['QPU_API_TOKEN'], 'config.json', require_complete=True)
train_and_analyze('config.json')
shot_noise_bootstrap('config.json', n_boot=500)
