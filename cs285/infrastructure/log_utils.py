import glob
import tensorflow as tf

DATA_FOLDER = '/home/harvey/Documents/cs285/CS285-Project/data/'

def get_section_results(file, X_label='Train_EnvstepsSoFar', Y_label='Train_AverageReturn'):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == X_label:
                X.append(v.simple_value)
            elif v.tag == Y_label:
                Y.append(v.simple_value)
    return X, Y

def getXY(exp_name, X_label='Train_EnvstepsSoFar', Y_label='Train_AverageReturn', X_max=float('inf')):
    logdir = (DATA_FOLDER+'%s/events*')%exp_name
    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results(eventfile, X_label=X_label, Y_label=Y_label)
    assert(len(X) == len(Y))
    X, Y = [x for x in X if x<X_max], [Y[i] for i in range(len(X)) if X[i]<X_max]
    return X, Y

def getXYpeer(exp_name, X_label='Train_EnvstepsSoFar', Y_label='Train_AverageReturn', peer_num=10, X_max=float('inf')):
    result = []
    for i in range(peer_num):
        Xi, Yi = getXY(exp_name, "Agent%d_"%i+X_label, "Agent%d_"%i+Y_label, X_max=X_max)
        if Xi and Yi:
            result.append((Xi, Yi))
    return result

def getXYensemble(exp_name, X_label='Train_EnvstepsSoFar', Y_label='Train_AverageReturn', num_agents=None, X_max=float('inf')):
    X, Y = getXY(exp_name, X_label, Y_label, X_max=X_max)
    if not num_agents:
        num_agents = int(exp_name[exp_name.find('ensemble') + 9])
    result = []
    for i in range(num_agents):
        Xi, Yi = X[i::num_agents], Y[i::num_agents]
        result.append((Xi, Yi))
    return result

def check(exp_name):
    X, Y = getXY(exp_name)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))