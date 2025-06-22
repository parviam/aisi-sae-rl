import warnings
warnings.filterwarnings('ignore')

import tqdm

import wandb
import numpy as np
from buffer import Buffer
from crosscoder import CrossCoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class CustomLearningRateSchedule:
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return np.float32(1.0)
        else:
            return np.float32(1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps))

    def get_learning_rate(self, initial_lr, global_step):
        learning_rate = initial_lr * tf.py_func(self.lr_lambda, [global_step], tf.float32)
        return learning_rate


class Trainer:
    def __init__(self, cfg, model_A, model_B):
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.crosscoder = CrossCoder(cfg)
        self.buffer = Buffer(cfg, model_A, model_B, verbose=True)
        self.total_steps = cfg["training_steps"]


        custom_lr_schedule = CustomLearningRateSchedule(self.total_steps)
        self.global_step = 0
        self.learning_rate = custom_lr_schedule.get_learning_rate(cfg["lr"], self.global_step)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=cfg["beta1"],
            beta2=cfg["beta2"]
        )
        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"])

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"]
        # over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.global_step < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.global_step \
                / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.global_step)
        print(loss_dict)

    def save(self):
        self.crosscoder.save()
        
    
    def step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.crosscoder.get_losses(inputs)
            loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        gradients = tape.gradient(loss, self.crosscoder.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(clipped_gradients, self.crosscoder.trainable_variables))
        
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        loss = loss.eval(session=sess)
        l0_loss = losses.l0_loss.eval(session=sess)
        l1_loss = losses.l1_loss.eval(session=sess)
        l2_loss = losses.l2_loss.eval(session=sess)
        explained_variance = losses.explained_variance.eval(session=sess)
        explained_variance_A = losses.explained_variance_A.eval(session=sess)
        explained_variance_B = losses.explained_variance_B.eval(session=sess)
        
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": l2_loss.item(),
            "l1_loss": l1_loss.item(),
            "l0_loss": l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "explained_variance": explained_variance.mean().item(),
            "explained_variance_A": explained_variance_A.mean().item(),
            "explained_variance_B": explained_variance_B.mean().item(),
        }
        return loss_dict

    def train(self):
        # Run the training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for i in tqdm.trange(self.total_steps):
                sess.run(self.learning_rate)
                acts = self.buffer.next()
                loss_dict = self.step(acts)
                self.global_step += 1
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()

            self.save()
