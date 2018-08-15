import models
import rl


def sample_dist(dist):
	return np.random.choice(np.arange(len(dist)), p=dist)