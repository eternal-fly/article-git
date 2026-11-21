import os
from experiment_common import dry_run, submit_jobs
dry_run('config.json')
submit_jobs(os.environ['QPU_API_TOKEN'], 'config.json')
