import matplotlib.pyplot as plt
import os
import glob
from tensorflow.compat.v1.train import summary_iterator
import numpy as np

def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value      
    return smoothed

#peer learning vs 1-agent baseline
folders = ["sac_peer_2_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_peer_5_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_ensemble_1_agents_1000_iter_HalfCheetah-v4_09-12-2022_23-23-06"]
colors = ["brown", "blue", "purple", "black"]
for folder in os.listdir('data'):
    if folder not in folders:
        continue
    logdir = 'data/{}/events*'.format(folder)
    for f in glob.glob(logdir):
        if "events.out" in f:
            eventfile = f
    Y = {}
    for e in summary_iterator(eventfile):
        for v in e.summary.value:
            if 'Eval_AverageReturn' in v.tag:
                if v.tag not in Y.keys():
                    Y[v.tag] = []
                Y[v.tag].append(v.simple_value)
    if "2_agents" in folder:
        Ys_2agents = list(Y.values())
    elif "3_agents" in folder:
        Ys_3agents = list(Y.values())
    elif "5_agents" in folder:
        Ys_5agents = list(Y.values())
    else:
        baselineY = Y["Eval_AverageReturn"]
plt.figure()
plt.tight_layout()
for i, Y in enumerate(Ys_2agents):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_2agents, axis=0)), 
            label="2 agent peer learning", color=colors[0])
for i, Y in enumerate(Ys_3agents):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[1])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_3agents, axis=0)), 
            label="3 agent peer learning", color=colors[1])
for i, Y in enumerate(Ys_5agents):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[2])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_5agents, axis=0)), 
            label="5 agent peer learning", color=colors[2])
plt.plot(range(0, 1000, 10), smooth(baselineY), '--',
            label="1 agent baseline", color=colors[3])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_peer_baseline_comparison.png", dpi=150)

#peer learning vs ensemble
folders = ["sac_peer_2_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_peer_5_agents_epsilon_0.6_adv_size_32_adv_dim_4_1000_iter_HalfCheetah-v4_11-12-2022_21-11-33",
           "sac_ensemble_2_agents_1000_iter_HalfCheetah-v4_09-12-2022_23-23-12",
           "sac_ensemble_3_agents_1000_iter_HalfCheetah-v4_09-12-2022_23-23-33",
           "sac_ensemble_5_agents_1000_iter_HalfCheetah-v4_10-12-2022_01-58-50"]
colors = ["blue", "black"]
for folder in os.listdir('data'):
    if folder not in folders:
        continue
    logdir = 'data/{}/events*'.format(folder)
    for f in glob.glob(logdir):
        if "events.out" in f:
            eventfile = f
    Y = {}
    for e in summary_iterator(eventfile):
        for v in e.summary.value:
            if 'Eval_AverageReturn' in v.tag:
                if v.tag not in Y.keys():
                    Y[v.tag] = []
                Y[v.tag].append(v.simple_value)
    if "peer_2_agents" in folder:
        Ys_2agents_peer = list(Y.values())
    elif "peer_3_agents" in folder:
        Ys_3agents_peer = list(Y.values())
    elif "peer_5_agents" in folder:
        Ys_5agents_peer = list(Y.values())
    elif "ensemble_2_agents" in folder:
        Ys_2agents_ensemble = list(Y.values())
    elif "ensemble_3_agents" in folder:
        Ys_3agents_ensemble = list(Y.values())
    elif "ensemble_5_agents" in folder:
        Ys_5agents_ensemble = list(Y.values())
###
plt.figure()
plt.tight_layout()
for i, Y in enumerate(Ys_2agents_peer):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_2agents_peer, axis=0)), 
            label="2 agent peer learning", color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_2agents_ensemble, axis=0)), 
            label="2 agent ensemble", color=colors[1])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_peer_ensemble_comparison_2.png", dpi=150)
###
plt.figure()
plt.tight_layout()
for i, Y in enumerate(Ys_3agents_peer):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_3agents_peer, axis=0)), 
            label="3 agent peer learning", color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_3agents_ensemble, axis=0)), 
            label="3 agent ensemble", color=colors[1])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_peer_ensemble_comparison_3.png", dpi=150)
###
plt.figure()
plt.tight_layout()
for i, Y in enumerate(Ys_5agents_peer):
    plt.plot(range(0, 1000, 10), smooth(Y), alpha=0.2, color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_5agents_peer, axis=0)), 
            label="5 agent peer learning", color=colors[0])
plt.plot(range(0, 1000, 10), smooth(np.mean(Ys_5agents_ensemble, axis=0)), 
            label="5 agent ensemble", color=colors[1])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_peer_ensemble_comparison_5.png", dpi=150)

#advice dim hyperparam variation
folders = ["sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-31-13",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_8_500_iter_HalfCheetah-v4_11-12-2022_22-37-37",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_16_500_iter_HalfCheetah-v4_11-12-2022_22-37-37"]
colors = ["brown", "blue", "purple"]
Ys = []
labels = []
for folder in os.listdir('data'):
    if folder not in folders:
        continue
    logdir = 'data/{}/events*'.format(folder)
    for f in glob.glob(logdir):
        if "events.out" in f:
            eventfile = f
    Y = {}
    for e in summary_iterator(eventfile):
        for v in e.summary.value:
            if 'Eval_AverageReturn' in v.tag:
                if v.tag not in Y.keys():
                    Y[v.tag] = []
                Y[v.tag].append(v.simple_value)
    Ys.append(list(Y.values()))
    labels.append(folder.split("32_")[1].split("_500")[0].replace("adv_dim_", "advice dim: "))
plt.figure()
plt.tight_layout()
for i, Ylist in enumerate(Ys):
    for Y in Ylist:
        plt.plot(range(0, 500, 10), smooth(Y), alpha=0.2, color=colors[i])
    plt.plot(range(0, 500, 10), smooth(np.mean(Ylist, axis=0)), 
                label="3 agent peer learning, " + labels[i], color=colors[i])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_advice_dim_variation.png", dpi=150)

#advice net size hyperparam variation
folders = ["sac_peer_3_agents_epsilon_0.6_adv_size_16_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_22-37-37",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-31-13",
           "sac_peer_3_agents_epsilon_0.6_adv_size_64_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_22-37-37"]
colors = ["brown", "blue", "purple"]
Ys = []
labels = []
for folder in os.listdir('data'):
    if folder not in folders:
        continue
    logdir = 'data/{}/events*'.format(folder)
    for f in glob.glob(logdir):
        if "events.out" in f:
            eventfile = f
    Y = {}
    for e in summary_iterator(eventfile):
        for v in e.summary.value:
            if 'Eval_AverageReturn' in v.tag:
                if v.tag not in Y.keys():
                    Y[v.tag] = []
                Y[v.tag].append(v.simple_value)
    Ys.append(list(Y.values()))
    labels.append(folder.split("0.6_")[1].split("_adv_dim")[0]
                    .replace("adv_size_", "advice net size: "))
plt.figure()
plt.tight_layout()
for i, Ylist in enumerate(Ys):
    for Y in Ylist:
        plt.plot(range(0, 500, 10), smooth(Y), alpha=0.2, color=colors[i])
    plt.plot(range(0, 500, 10), smooth(np.mean(Ylist, axis=0)), 
                label="3 agent peer learning, " + labels[i], color=colors[i])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_advice_net_size_variation.png", dpi=150)

#epsilon hyperparam variation
folders = ["sac_peer_3_agents_epsilon_0_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-12-22",
           "sac_peer_3_agents_epsilon_0.1_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-31-13",
           "sac_peer_3_agents_epsilon_0.3_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-31-13",
           "sac_peer_3_agents_epsilon_0.6_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_18-31-13",
           "sac_peer_3_agents_epsilon_1_adv_size_32_adv_dim_4_500_iter_HalfCheetah-v4_11-12-2022_17-35-38"]
colors = ["black", "brown", "blue", "purple", "red"]
Ys = []
labels = []
for folder in os.listdir('data'):
    if folder not in folders:
        continue
    logdir = 'data/{}/events*'.format(folder)
    for f in glob.glob(logdir):
        if "events.out" in f:
            eventfile = f
    Y = {}
    for e in summary_iterator(eventfile):
        for v in e.summary.value:
            if 'Eval_AverageReturn' in v.tag:
                if v.tag not in Y.keys():
                    Y[v.tag] = []
                Y[v.tag].append(v.simple_value)
    Ys.append(list(Y.values()))
    labels.append(folder.split("agents_")[1].split("_adv_size")[0]
                    .replace("epsilon_", "$\epsilon$: "))
plt.figure()
plt.tight_layout()
for i, Ylist in enumerate(Ys):
    for Y in Ylist:
        plt.plot(range(0, 500, 10), smooth(Y), alpha=0.2, color=colors[i])
    plt.plot(range(0, 500, 10), smooth(np.mean(Ylist, axis=0)), 
                label="3 agent peer learning, " + labels[i], color=colors[i])
plt.legend(loc="lower right")
plt.xlabel("Iteration")
plt.ylabel("Eval_AverageReturn")
plt.savefig("value_signals_epsilon_variation.png", dpi=150)