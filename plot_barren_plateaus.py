import pickle
path="variances_nlayers_1_2_3_plaquettes_1_2_3_4_samples_100.pkl"
with open(path, 'rb') as f:
    plaq4 = pickle.load(f)

variances_dict=plaq4

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # Optional: LaTeX default font
}) 

# Define markers and colors
layer_markers = ["o", "s", "D", "^", "v", "<", ">"]  # Extend if needed
colors = {"Invariant": "b", "mbqc":"g"}
names={"Invariant": "GI", "mbqc":"ZZ"}
qubits = np.array([8, 13, 18, 23])

plt.figure(figsize=(6, 4))
#plt.rcParams.update({'font.size': 14})

plt.rc('font', size=10) #controls default text size
#plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=16) #fontsize of the x and y labels
plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
plt.rc('ytick', labelsize=16) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend


for ansatz_type in ["Invariant", "mbqc"]:
    for i, (n_layers, variances) in enumerate(variances_dict[ansatz_type].items()):
        marker = layer_markers[i % len(layer_markers)]  # Cycle through markers
        #p = np.polyfit(qubits, np.log(variances), 1)
        if ansatz_type!="hea":
            
            plt.semilogy(
                qubits, variances, marker=marker, linestyle="-", color=colors[ansatz_type],
                label=f"{ansatz_type} - {n_layers} layers"
            )
            plt.xticks(qubits)
        else:
            qubitsh = np.array([2 + 6 * n_p for n_p in [1,2,3]])
            plt.semilogy(
                qubitsh, variances, marker=marker, linestyle="-", color=colors[ansatz_type],
                label=f"{ansatz_type} - {n_layers} layers"
            )
            plt.xticks(qubits)

plt.ylim(1e-2,14)
plt.xlabel("Qubits")
plt.ylabel(r"$\langle \partial \theta_{N} C\rangle$ Variance")

# Create custom legend handles
color_legend = [Line2D([0], [0], color=colors[at], lw=4, label=names[at]) for at in colors]
marker_legend = [
    Line2D([0], [0], marker=layer_markers[i % len(layer_markers)], linestyle="", color="black", label=f"{n_layers} layers")
    for i, n_layers in enumerate(sorted({nl for ansatz in variances_dict.values() for nl in ansatz}))
]

# Add legends
legend2 = plt.legend(handles=marker_legend, loc="lower right")
legend1 = plt.legend(handles=color_legend, loc="lower left")

# Add both legends to the plot
plt.gca().add_artist(legend2)
plt.tight_layout()
plt.show()

