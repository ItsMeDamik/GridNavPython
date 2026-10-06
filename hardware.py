import vivilux

#from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import ThrMSE, ThrSSE

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
import pathlib
from os import path
import json

parser = argparse.ArgumentParser(description='Run a neuromorphic error-driven learning example with idealized hardware.')
parser.add_argument("-s", '--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument("-n", '--numEpochs', type=int, default=50, help='Number of training epochs')
args = parser.parse_args()

np.random.seed(seed=args.seed)

numEpochs = args.numEpochs
inputSize = 4
hiddenSize = 4
outputSize = 2
