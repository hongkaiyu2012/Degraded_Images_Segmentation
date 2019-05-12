#!/usr/bin/env python

import os
import pytz
import yaml
import shlex
import datetime
import subprocess
import os.path as osp


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')

    # create out
    here = osp.dirname(osp.abspath(__file__))
    # go to its parent folder
    here = osp.dirname(here)
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_log_test_dir(model_name, dataset, degradedtest, test_model):
    name = 'Test-%s-%s' % (model_name, dataset)
    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    nowtime = 'Degradation-%s_TIME-%s' % (degradedtest, now.strftime('%Y%m%d-%H%M%S'))

    # creat out
    here = osp.dirname(osp.abspath(__file__))
    # go to its parent folder
    here = osp.dirname(here)
    log_dir = osp.join(here, 'logs', name, nowtime)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'test_model.txt'), 'w') as f:
        f.write('Model: ' + test_model + '\n')
    return log_dir