import numpy as np
from Utils import get_born_state
from model_class import Model


# Model based epsilon-greedy exploration (MBEG)
# eps + alpha < 1
def mb_eps_greedy(mdp, budget, eps=0.1, horizons=None, pi_external=None, alpha=0):
    # Create a model object with T_hat, R_hat etc
    model = Model(mdp)
    # Init agent state
    cur_state = get_born_state(mdp.born, mdp.nstates)
    # Data points
    data = []
    # flag = True
    pi_optimal = np.random.randint(0, model.nactions, model.nstates)
    for t in range(1, budget + 1):
        # print("Current iteration {0}".format(t))
        # Choose action as per MBEG algorithm
        rand = np.random.uniform(0.0, 1.0)
        if model.model_valid:
            # MDP plan for optimal policy under current model
            values, pi_optimal = model.plan()
            if rand < eps:
                action = np.random.randint(0, mdp.nactions)
                # print("Random action {0} chosen".format(action))
            elif eps < rand < eps + alpha:
                action = pi_external[cur_state]
            else:
                action = pi_optimal[cur_state]
                # print("Greedy action {0} chosen".format(action))
        else:
            if rand < alpha:
                action = pi_external[cur_state]
            else:
                action = np.random.randint(0, mdp.nactions)
            # print("Random action {0} chosen".format(action))
        # Query the MDP sample model
        rew, next_state, epi_ended = mdp.sample(cur_state, action)
        model.update(cur_state, action, rew, next_state)
        # print("Next state is {0}".format(next_state))
        cur_state = next_state
        # Be reborn if episode ended
        if epi_ended:
            cur_state = get_born_state(mdp.born, mdp.nstates)
        if t in horizons:
            true_value = mdp.evaluate_policy(pi_optimal)
            data.append(true_value)
    # print(model.T_hat)
    # print(model.R_hat)
    # Return optimal policy as per learned model, if model_valid
    if model.model_valid:
        return data, model.plan()
    else:
        print("Exploration unsuccessful, model not learnt")
        exit(-1)


# Model based action elimination, an eps-delta PAC-RL algorithm
# Even-Dar et al. 2004, 2006, JMLR
def mb_action_elimination(mdp, budget, eps, delta):
    # Create a model object with T_hat, R_hat etc
    model = Model(mdp)
    # Init agent state
    cur_state = get_born_state(mdp.born, mdp.nstates)
    Rmax = model.rmax
    Vmax = Rmax/(1 - model.gamma)
    # Init set of viable state action-pairs
    Uo = []
    for s in range(model.nstates):
        Uo.append([])
    for s in range(model.nstates):
        for a in range(model.nactions):
            Uo[s].append(a)
    # Array to hold exploration bonus terms for every (s, a)
    explore_term = np.zeros((model.nstates, model.nactions), dtype=np.float32)
    # Initilaise seed policies given to policy iteration algorithm
    pi_upper = np.random.randint(0, model.nactions, model.nstates)
    pi_lower = np.copy(pi_upper)
    # Flag to continue
    converged = False
    for t in range(1, budget + 1):
        print("Current iteration {0}".format(t))
        if model.model_valid:
            # Choose any random viable action as per MBAE algorithm
            action = np.random.choice(Uo[cur_state])
            # Confidence interval upper/lower bound term for MBAE
            explore_term = Vmax*np.sqrt(np.log(1*t*t*model.nstates*model.nactions/delta)/model.total_visits)
            # Get V_UCB, upper confidence bound on values
            values_upper, pi_upper = model.plan(init_policy=pi_upper, pac_type='MBAE', explore_terms=explore_term)
            # Get V_LCB, lower confidence bound on values, same as above except explore terms are -ve
            values_lower, pi_lower = model.plan(init_policy=pi_lower, pac_type='MBAE', explore_terms=-explore_term)
            Q_upper = np.sum(model.T_hat * model.R_hat, axis=2) + \
                      model.gamma * np.sum(model.T_hat * values_upper, axis=2) + explore_term
            Q_lower = np.sum(model.T_hat * model.R_hat, axis=2) + \
                      model.gamma * np.sum(model.T_hat * values_lower, axis=2) - explore_term
            # Eliminate sub-optimal actions
            for s in range(model.nstates):
                for a in Uo[s]:
                    if Q_upper[s, a] < values_lower[s]:
                        Uo[s].remove(a)
            # Assume converged
            converged = True
            for s in range(model.nstates):
                for a in Uo[s]:
                    print(Q_upper[s, a], Q_lower[s, a])
                    if abs(Q_upper[s, a] - Q_lower[s, a]) > eps * (1 - model.gamma) / 2:
                        converged = False
                        break
            if converged:
                print("Converged")
                break
        else:
            action = np.random.randint(0, mdp.nactions)
        # Query the MDP sample model
        rew, next_state, epi_ended = mdp.sample(cur_state, action)
        # Update agent's model of MDP
        model.update(cur_state, action, rew, next_state)
        # print("Next state is {0}".format(next_state))
        cur_state = next_state
        # Be reborn if episode ended
        if epi_ended:
            cur_state = get_born_state(mdp.born, mdp.nstates)
    # print(model.T_hat)
    # print(model.R_hat)
    print(Uo)
    # Return optimal policy as per learned model, if model_valid
    if model.model_valid:
        return np.argmax(Q_lower, axis=1)
    else:
        print("Exploration unsuccessful, model not learnt")
        exit(-1)
