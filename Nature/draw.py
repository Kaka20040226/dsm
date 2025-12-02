import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(data):
    window = 50
    data = np.array(data)
    # smooth
    for i in range(len(data)):
        data[i] = np.mean(data[max(0,i-window):min(len(data),i+window)])
    return data

data = []
with open("draw1.json", "r") as f:
    data = json.load(f)

# 4 subplots
fig, axs = plt.subplots(2, 2)

for key in data.keys():
    data[key] = [smooth(data[key][0]),smooth(data[key][1]),smooth(data[key][2]),smooth(data[key][3])]
    axs[0, 0].plot(list(range(len(data[key][0]))),data[key][0],  label=key)
    axs[0, 0].set_title("Loss")
    axs[1, 1].plot(list(range(len(data[key][1]))), data[key][1], label=key)
    axs[1, 1].set_title("Objective Value 2")
    axs[1, 1].set_ylim([0.465, 0.475])
    axs[0, 1].plot(list(range(len(data[key][2]))), data[key][2], label=key)
    axs[0, 1].set_title("Objective Value 1")
    axs[1, 0].plot(list(range(len(data[key][3]))), data[key][3], label=key)
    axs[1, 0].set_title("Crtical Path Length")

plt.show()
    
    