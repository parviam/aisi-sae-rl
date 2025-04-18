import argparse
import json
import sys

import yaml
import gym
from baselines.common.models import build_impala_cnn
import tensorflow as tf
from baselines.ppo2.model import Model
from trainer import Trainer
from procgen import ProcgenEnv
from baselines.common.policies import build_policy
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from mpi4py import MPI

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out

class ModifiedModel(Model):
    def __init__(self, layer_name, *args, **kwargs):
        super(ModifiedModel, self).__init__(*args, **kwargs)
        self.layer_name = layer_name

    def build(self, obs):
        x = super(ModifiedModel, self).build(obs)
        # Get the output of the specified layer
        for op in tf.get_default_graph().get_operations():
            if op.name == self.layer_name:
                return op.outputs[0]
        raise ValueError(f"Layer {self.layer_name} not found")

def create_ppo_model(network, nsteps=256, nminibatches=8):
    venv = ProcgenEnv(num_envs=1,
        env_name='coinrun',
        num_levels=1000,
        start_level=0,
        distribution_mode='hard')
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    policy = build_policy(venv, network)
    nenvs = venv.num_envs
    ob_space = venv.observation_space
    ac_space = venv.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    comm = MPI.COMM_WORLD

    model = ModifiedModel('ppo2_model/pi/layer_15/kernel:0', policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef = .01, vf_coef=0.5,
                    max_grad_norm=0.5, comm=comm, mpi_rank_weight=1)
    
    return model


def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments,
    convert these to command line arguments,
    look at what was passed in, and return an updated dictionary.
    If in Ipython, just returns with no changes
    """
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) is bool:
            # argparse for Booleans is broken rip.
            # Now you put in a flag to change the default --{flag} to set True,
            # --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


with open("config.yaml", "r") as file:
    default_cfg = yaml.safe_load(file)

load_path = '/data1/projects/keaton_gt_research/aisi-sae-rl/train-procgen/train_procgen/p-adelarue3-0/default_experiment/checkpoints/00001'
# with tf.variable_scope("modelA"):
model_A = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
model_A_wrapped = create_ppo_model(model_A)
model_A_wrapped.load(load_path)
# model_B = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
# model_B_wrapped = create_ppo_model(model_B)
# model_B_wrapped.load(load_path)
    
exit()

cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, model_A_wrapped, model_B_wrapped)
trainer.train()
