# How do RL Agents Evolve? 
### (unordered) Ryan Bowers, Parv Mahajan, Rohan Mehta, Kenneth Eaton, Abhay Sheshadri

#### A capstone project of the [AI Safety Institute](https://www.aisi.dev) Foundation Fellowship at the Georgia Institute of Technology

## Project Description
We plan to train a policy using RL/behavior cloning on some environment, caching intermediate activations at each gradient step over time.  
Then, we’ll train a sparse autoencoder to reconstruct the activations-over-time.  
We hope to be able to find latents that we can identify as primitive early-training features, or more complex late-stage features.  

We also hope to be able to detect features related to unsafe trajectories, such that we can minimize harms during deployment. 
We hope to quantify and explore the randomness inherent to DeepRL training, and determine whether training features conserve. 
That is to say, we hope to explore whether identically-trained policies are semantically consistent. 
We hope to identify differences in learned features between model architectures like Policy Gradient, Actor-Critic, Q-Learning, and more and 
training styles like online, offline, and behavior cloning.

## Expected Goals
- We hope to learn about the use of SAEs as explainability tools
- We hope to contribute new insights about how RL models train and learn over time in complex environments.
- We hope to learn how to efficiently and harmlessly train DeepRL systems
- We hope to contribute a large, open dataset with the activations of our RL model for other researchers to study and manipulate.

## Setup
To setup the conda environment with the required python packages, use environment.yml like:

    conda env create -n rl_env --file=environment.yml