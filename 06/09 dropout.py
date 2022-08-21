import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.mnist import load_mnist
from optimizer import SGD
from funcs import *
from networks import MultiLayerNet