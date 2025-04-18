import json
import pprint
from pathlib import Path
from typing import NamedTuple, Optional, Union

import einops
import tensorflow as tf


DTYPES = {"fp32": tf.float32, "fp16": tf.float16, "bf16": tf.bfloat16}
SAVE_DIR = Path("crosscoder/checkpoints")


class LossOutput(NamedTuple):
    l2_loss: tf.Tensor
    l1_loss: tf.Tensor
    l0_loss: tf.Tensor
    explained_variance: tf.Tensor
    explained_variance_A: tf.Tensor
    explained_variance_B: tf.Tensor


class CrossCoder(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        # self.dtype = DTYPES[self.cfg["enc_dtype"]]
        tf.set_random_seed(self.cfg["seed"])
        # hardcoding n_models to 2
        self.W_enc = tf.Variable(tf.random.normal([2, d_in, d_hidden]))
        self.W_dec = tf.Variable(tf.random.normal([d_hidden, 2, d_in]))
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec = self.W_dec / tf.norm(self.W_dec, axis=-1, keepdims=True) * self.cfg["dec_init_norm"]
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc = tf.transpose(self.W_dec, [2, 1, 0])
        self.b_enc = tf.Variable(tf.zeros([d_hidden]))
        self.b_dec = tf.Variable(tf.zeros([2, d_in]))
        self.d_hidden = d_hidden

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = tf.einsum('ijk,klm->ijm', x, self.W_enc)
        if apply_relu:
            acts = tf.nn.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = tf.einsum('ij,jk->ik', acts, self.W_dec)
        return acts_dec + self.b_dec

    def call(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        # x = tf.cast(x, self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct - x
        squared_diff = diff ** 2
        l2_per_batch = tf.reduce_sum(squared_diff, axis=[1, 2])
        l2_loss = tf.reduce_mean(l2_per_batch)
        total_variance = tf.reduce_sum((x - tf.reduce_mean(x, axis=0)) ** 2, axis=[0, 1, 2])
        explained_variance = 1 - l2_per_batch / total_variance
        per_token_l2_loss_A = tf.reduce_sum((x_reconstruct[:, 0, :] - x[:, 0, :]) ** 2, axis=-1)
        total_variance_A = tf.reduce_sum((x[:, 0, :] - tf.reduce_mean(x[:, 0, :])) ** 2, axis=-1)
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
        per_token_l2_loss_B = tf.reduce_sum((x_reconstruct[:, 1, :] - x[:, 1, :]) ** 2, axis=-1)
        total_variance_B = tf.reduce_sum((x[:, 1, :] - tf.reduce_mean(x[:, 1, :])) ** 2, axis=-1)
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B
        decoder_norms = tf.norm(self.W_dec, axis=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = tf.reduce_sum(decoder_norms, axis=-1)
        l1_loss = tf.reduce_mean(acts * total_decoder_norm[:, None])
        l0_loss = tf.reduce_mean(tf.cast(acts > 0, tf.float32))
        return LossOutput(
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            l0_loss=l0_loss,
            explained_variance=explained_variance,
            explained_variance_A=explained_variance_A,
            explained_variance_B=explained_variance_B,
        )

    def create_save_dir(self):
        version_list = [
            int(file.name.split("_")[1])
            for file in list(SAVE_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = SAVE_DIR / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.h5"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"
        self.save_weights(weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1


    @classmethod
    def load(cls, version_dir, checkpoint_version):
        save_dir = SAVE_DIR / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.h5"
        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_weights(weight_path)
        return self
