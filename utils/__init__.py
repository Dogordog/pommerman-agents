import numpy as np

from . import models
from . import rl


def sample_dist(dist):
	return np.random.choice(np.arange(len(dist)), p=dist)