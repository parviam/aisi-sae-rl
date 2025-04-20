import warnings
warnings.filterwarnings('ignore')

import functools
import argparse
import json
import sys
import multiprocessing
import yaml
import gym
# from baselines.common.models import build_impala_cnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from baseline_utils import Model, get_session, build_policy, load_variables, build_impala_cnn
from trainer import Trainer
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from mpi4py import MPI



def create_ppo_model(network, model_name, nsteps=256, nminibatches=8):
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

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=.01, vf_coef=0.5,
                    max_grad_norm=0.5, model_name=model_name, comm=comm, mpi_rank_weight=1)
    
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

load_path = '/data1/projects/keaton_gt_research/aisi-sae-rl/train-procgen/train_procgen/p-adelarue3-0/restore/checkpoints/00001'
# with tf.variable_scope("modelA"):
# with tf.name_scope("model1"):
# model_A = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
# model_A_wrapped = create_ppo_model(model_A)
# model_A_wrapped.load(load_path)

# trainable_vars = tf.trainable_variables()
# tf.get_default_graph().clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# trainable_vars = tf.trainable_variables()


# Try to rename variables
# trainable_variables = tf.trainable_variables()

# new_variable_names = []
# for var in trainable_variables:
#     new_name = var.name.replace("layer", "layerA").split(":")[0] # Example: Adding "my_dense_" to dense layers
#     new_variable_names.append(new_name)

# with tf.name_scope("model1"):
#     new_trainable_variables = []
#     for i, var in enumerate(trainable_variables):
#         # new_name = new_variable_names[i]
#         # new_variable = tf.Variable(var.value(), name=new_name, trainable=True)
#         # new_trainable_variables.append(new_variable)
#         with tf.Session() as sess:
#             value = var.value().eval()
#             print(value)

# with tf.name_scope("model1"):
# new_model.scope = tf.get_variable_scope()
# for i, var in enumerate(new_trainable_variables):
#     tf.assign(trainable_variables[i], var).eval()
#     model_A_wrapped.scope.reuse_variables()
    

#  Update model's internal state or create a new model as needed
#  (This part depends on how build_impala_cnn is structured)

def convert_to_keras(orig_model):
    inputs = tf.keras.Input(shape=(64, 64, 3))
    outputs = orig_model.act_model.policy_network(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    layers = model.layers

    # Find the layer that corresponds to 'cnn_out'
    for i, layer in enumerate(layers):
        if 'flatten' in layer.name:
            cnn_out = layer.output
            break

    cnn_out_model = tf.keras.Model(inputs=model.input, outputs=cnn_out)
    return cnn_out_model
        
sess = sess = get_session()
load_ckpt = functools.partial(load_variables, sess=sess)


model_name1 = "modelA"
with tf.variable_scope(model_name1):
    model_A = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    model_A_wrapped = create_ppo_model(model_A, model_name1)
    load_ckpt(load_path, f"{model_name1}/")
    

model_name2 = "modelB"
with tf.variable_scope(model_name2):
    model_B = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    model_B_wrapped = create_ppo_model(model_B, model_name2)
    load_ckpt(load_path, f"{model_name2}/")

model_A_keras = convert_to_keras(model_A_wrapped)
model_B_keras = convert_to_keras(model_B_wrapped)
    

cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, model_A_keras, model_B_keras)
trainer.train()


# Use the new model to get the 'cnn_out' value
unscaled_images = tf.random.normal([1, 64, 64, 3])
cnn_out_value = model_A_keras(unscaled_images)
print(cnn_out_value)
