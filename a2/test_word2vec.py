
import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax
from word2vec import sigmoid

########## sigmoid check ###############
x = np.asarray([0,2,5])
print(sigmoid(x))
########################################
