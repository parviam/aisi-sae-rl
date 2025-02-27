# Impala + Coinrun Progress

# Why is this hard?

tldr - Refer to `training_pipeline/Impala_v2.ipynb` to see how to train Impala on Procgen environment. A couple of key things to call out

*Procgen Coinrun is build for OpenAI Gym v21. Stable baselines expects a gymnasium environment.*

> Okay so just use shimmy which wraps older Gym environments to work with Gymnasium API

*When you wrap it in shimmy, the stable baselines API expects to be able to reset the environment with a specific seed.*

> Create a custom wrapper environment that will call all the appropriate functions in the child. Add a function for reset that takes a seed but doesn't actually use it

*Stable baselines doesn't come with an implementation of Impala. And OpenAI's original codebase uses Tensorflow.*

> Write custom model based on the PyTorch implementation. Supply that architecture to Stable Baselines*

*I don't think its using my implementation. When I try to run my model with a print statement, its never executed. I am following the [directions](https://stable-baselines3.readthedocs.io/en/sde/guide/custom_policy.html) exactly!*

> Yeah the docs are really bad. What's happening is that the ActorCriticPolicy you are importing from has a lot of special conditions running in the `__init__` which is causing the issue. To solve this, carefully read through the [implementation](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L416) and override the appropriate parameters. Specifically, you will have to set the `share_features_extractor=True`, `net_arch` so that sb3 knows what the policy and value function needs in terms of hidden layers. In addition, ensure you point sb3 to your custom ImpalaCNN feature extrator model using `make_feature_extractor(self)`.

*I trained the model and running it in eval mode and it works, but when I try loading the saved model its not working.*

> This is some weirdness with path saving and sb3. Ensure you are loading via `model_loaded = PPO.load(PATH, env=wrapped_env, verbose=True, policy=ImpalaActorCriticPolicy)` and saving to the absolute path and not the sym link (not `~/p-adelarue3-0/` but USE `/storage/coda1/p-adelarue3/0/rmehta98/impala_sanity`)

*It is only running on the CPU*

> PENDING SOLUTION


# Experiment 1
> Sanity Test - Does Impala overfit to 1 level?
```python
coin_non_env = GymV21CompatibilityV0("procgen:procgen-coinrun-v0", make_kwargs={
        "num_levels": 1,
        "start_level": 0,
        "distribution_mode": "easy"
    })
```

This was successful. Saved in Rohan's adelarue3 folder. Around ~100,000 the model overfitted and was consistently scoring 9.5+ / 10 on the coinrun environment. Note, this is only for 1 level for the purposes of testing whether the training loop works. The next will assess how well the model is generalizing. 