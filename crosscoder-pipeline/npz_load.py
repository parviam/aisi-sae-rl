from numpy import load

"""
Basic script to load and view the npz file of states and model activations
Format:
states (num_states, 64, 64, 3)
activations_A (num_states, num_nodes)
activations_B (num_states, num_nodes)

"""

data = load('test_state_acts.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item].shape)