import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

experiment_ID="experiment_1B"

with open('./logs/scalars/experiment_1B/stiffness_v0/babbling_time_1_mins/mc_run0/trainHistoryDict.pickle', 'rb') as f:
	history = pickle.load(f)
#import pdb; pdb.set_trace()


# Plot training & validation accuracy values


# Plot training & validation loss values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()