import tqdm

import wandb
from buffer import Buffer
from crosscoder import CrossCoder
import tensorflow as tf

class Trainer:
    def __init__(self, cfg, model_A, model_B):
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.crosscoder = CrossCoder(cfg)
        self.buffer = Buffer(cfg, model_A, model_B)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]


        # Create the learning rate schedule
        self.scheduler = tf.train.LearningRateScheduler(
            lambda global_step: lr_lambda(global_step) * cfg["lr"]
        )
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr_placeholder,
            beta1=cfg["beta1"],
            beta2=cfg["beta2"]
        )
        
        self.step_counter = 0
        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"])

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2
                                                            * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"]
        # over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter \
                / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save()
        
    
    def step(inputs):
        with tf.GradientTape() as tape:
            losses = self.crosscoder.get_losses(inputs)
            loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        gradients = tape.gradient(loss, self.crosscoder.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(clipped_gradients, self.crosscoder.trainable_variables))
        
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        return loss_dict

    def train(self):
        self.step_counter = 0

        # Run the training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for i in tqdm.trange(self.total_steps):
                lr = self.scheduler(self.step_counter)
                acts = self.buffer.next()
                loss_dict = step(acts)
                self.step_counter += 1
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()

            self.save()
