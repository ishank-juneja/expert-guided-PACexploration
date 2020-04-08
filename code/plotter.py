import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import argparse
import pathlib

plt.plot(horizons, eps_array_expert, color='b')
plt.plot(horizons, eps_array_vanilla, color='r')
plt.xlabel("Horizon - T")
plt.ylabel("Diff. b/w optimal and learned policy")
plt.title("Effect of external policy on exploration")
plt.legend(["With Expert", "Without Expert"])
plt.savefig("../figures/mdp10.png")
plt.show()
plt.close()
mean_horizon = np.zeros((nseeds, len(alphas)), dtype=np.float32)
for rs in range(nseeds):
    print("Random Seed = {0}".format(rs))
    for it, alpha in enumerate(alphas):
        random.seed(rs)
        np.random.seed(rs)
        horizon, (v_hat1, pi_hat1) = MBEG(mdp, 500000, eps=0.1, alpha=alpha, pi_external=pi_external)
        mean_horizon[rs, it] += horizon
mean_horizon = np.mean(mean_horizon, axis=0)
plt.loglog(alphas, mean_horizon, color='b')
plt.xlabel("External Policy trust probability")
# plt.xticks(alphas)
plt.ylabel("Horizon to learn optimal policy")
plt.title("Effect of external policy on exploration")
plt.savefig("../figures/riverSwim.png")
plt.show()
plt.close()