import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import functools
import multiprocessing
import os
import os.path as osp
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.a2c.utils import fc
from baselines.common import explained_variance, set_global_seeds, tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.input import encode_observation, observation_placeholder
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from baselines.common.mpi_util import sync_from_root
from baselines.common.tf_util import (adjust_shape, initialize, save_variables)
from baselines.ppo2.ppo2 import safemean
from baselines.ppo2.runner import Runner
from mpi4py import MPI


def build_impala_cnn(unscaled_images, depths=[16, 32, 32], **conv_kwargs):
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
        return tf.keras.layers.Conv2D(
            depth, 3, padding="same", name="layer_" + get_layer_num_str()
        )(out)

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(out)
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.0

    for depth in depths:
        out = conv_sequence(out, depth)

    cnn_out = tf.keras.layers.Flatten()(out)
    out = tf.nn.relu(cnn_out)
    out = tf.keras.layers.Dense(
        256, activation=tf.nn.relu, name="layer_" + get_layer_num_str()
    )(out)

    return out


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
        )
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


def load_variables(load_path, model_name, variables=None, sess=None):
    import joblib

    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    for v in variables:
        if model_name in v.name:
            lookup_name = model_name + v.name.split(model_name)[-1]
            if lookup_name not in loaded_params:
                # edge case for beta1 and beta2
                lookup_name = v.name.split(model_name)[-1][1:]
            restores.append(v.assign(loaded_params[lookup_name]))

    sess.run(restores)


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """

    def __init__(
        self,
        *,
        policy,
        ob_space,
        ac_space,
        nbatch_act,
        nbatch_train,
        nsteps,
        ent_coef,
        vf_coef,
        max_grad_norm,
        model_name,
        mpi_rank_weight=1,
        comm=None,
        microbatch_size=None
    ):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope("ppo2_model", reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(
            train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE
        )
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))
        )

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables(model_name)
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(
                comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5
            )
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm)  # pylint: disable=E1101

    def train(
        self,
        lr,
        cliprange,
        obs,
        returns,
        masks,
        actions,
        values,
        neglogpacs,
        states=None,
    ):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values,
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]

def constfn(val):
    def f(_):
        return val
    return f

def learn(
    *,
    network,
    env,
    total_timesteps,
    model_name,
    eval_env=None,
    seed=None,
    nsteps=2048,
    ent_coef=0.0,
    lr=3e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    nminibatches=4,
    noptepochs=4,
    cliprange=0.2,
    save_interval=0,
    load_path=None,
    model_fn=None,
    update_fn=None,
    init_fn=None,
    mpi_rank_weight=1,
    comm=None,
    **network_kwargs
):
    """
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    """

    set_global_seeds(seed)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = MPI is None or MPI.COMM_WORLD.Get_rank() == 0

    model = Model(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        model_name=model_name,
        comm=comm,
        mpi_rank_weight=mpi_rank_weight,
    )

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(
            env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam
        )

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root:
            logger.info("Stepping environment...")

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = (
            runner.run()
        )  # pylint: disable=E0632
        if eval_env is not None:
            (
                eval_obs,
                eval_returns,
                eval_masks,
                eval_actions,
                eval_values,
                eval_neglogpacs,
                eval_states,
                eval_epinfos,
            ) = eval_runner.run()  # pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root:
            logger.info("Done.")

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None:  # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (
                        arr[mbinds]
                        for arr in (obs, returns, masks, actions, values, neglogpacs)
                    )
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (
                        arr[mbflatinds]
                        for arr in (obs, returns, masks, actions, values, neglogpacs)
                    )
                    mbstates = states[mbenvinds]
                    mblossvals.append(
                        model.train(lrnow, cliprangenow, *slices, mbstates)
                    )

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update * nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]))
            logger.logkv("eplenmean", safemean([epinfo["l"] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv(
                    "eval_eprewmean",
                    safemean([epinfo["r"] for epinfo in eval_epinfobuf]),
                )
                logger.logkv(
                    "eval_eplenmean",
                    safemean([epinfo["l"] for epinfo in eval_epinfobuf]),
                )
            logger.logkv("misc/time_elapsed", tnow - tfirststart)
            for lossval, lossname in zip(lossvals, model.loss_names):
                logger.logkv("loss/" + lossname, lossval)

            logger.dumpkvs()
        print(f">> {update} % {save_interval} == {update % save_interval} \t {logger.get_dir()} \t {is_mpi_root}")
        if (
            save_interval
            and (update % save_interval == 0 or update == 1)
            and logger.get_dir()
            and is_mpi_root
        ):
            checkdir = osp.join(logger.get_dir(), "checkpoints")
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, "%.5i" % update)
            print("Saving to", savepath)
            model.save(savepath)


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(
        self,
        env,
        observations,
        latent,
        policy_network,
        estimate_q=False,
        vf_latent=None,
        sess=None,
        **tensors
    ):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        self.policy_network = policy_network
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, "q", env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, "vf", 1)
            self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == "Placeholder":
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp], observation, **extra_feed
        )
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def build_policy(
    env,
    policy_network,
    value_network=None,
    normalize_observations=False,
    estimate_q=False,
    **policy_kwargs
):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = (
            observ_placeholder
            if observ_placeholder is not None
            else observation_placeholder(ob_space, batch_size=nbatch)
        )

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors["rms"] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope("pi", reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert (
                        nenv > 0
                    ), "Bad input for recurrent policy: batch size {} smaller than nsteps {}".format(
                        nbatch, nsteps
                    )
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == "shared":
            vf_latent = policy_latent
        else:
            if _v_net == "copy":
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope("vf", reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            policy_network=policy_network,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn
