#include "sarsa.h"
Sarsa::Sarsa() {
}

Sarsa::Sarsa(int numStates, int numActions, int L) {
    // Get the number of states and actions that have actually been observed
    this->numStates = numStates;
    this->numActions = numActions;
    this->L = L;

    // Resize everything and set all to zero
    d0.resize(numStates);
    Q.resize(L);
    for (int t = L-1; t >= 0; t--) {
        Q[t] = MatrixXd::Zero(numStates, numActions);
    }
    actionProbabilities.resize(numStates);
}


void Sarsa::loadEvalPolicy(const vector<Trajectory*> & trajs,
                           Policy evalPolicy) {
    unsigned int N = trajs.size();
    for (int s=0; s < numStates; s++)
        d0[s] = 0.0;

    for (int s = 0; s < numStates; s++) {
        actionProbabilities[s] = evalPolicy.getActionProbabilities(s);
    }

    if (N > 0) {
        // Compute d0
        for (unsigned int i = 0; i < N; i++)
            d0[trajs[i]->states[0]] += 1.0 / static_cast<double>(N);

        double alpha = 0.5;
        int n_iters = 100;
        int itr = 0;
        int s, a, sPrime;
        double R;
        double target;
        double delta = 0.0;
        double eps = 1.e-6;
        // TODO(jphanna): run until convergence instead of a fixed number
        // of iterations.
        do {
            delta = 0.0;
            for (auto &traj : trajs) {
                for (int t=traj->len - 1; t >= 0; t--) {
                    s = traj->states[t];
                    a = traj->actions[t];
                    R = traj->rewards[t];
                    if (t < traj->len - 1) {
                        sPrime = traj->states[t + 1];
                        target = R +
                          actionProbabilities[sPrime].dot(Q[t+1].row(sPrime));
                    } else {
                        target = R;
                    }
                    Q[t](s, a) += alpha * (target - Q[t](s, a));
                    if (alpha * abs(target - Q[t](s, a)) > delta)
                        delta = alpha * abs(target - Q[t](s, a));
                }
            }
            itr++;
        } while (delta > eps);
        // printf("%d %f\n", itr, delta);
    }
    // std::cout <<     actionProbabilities[2] << std::endl;
    // std::vector<double> v(numStates);
    // for (int s=0; s < numStates; s++) {
    //     v[s] = 0.0;
    //     for (int a=0; a < numActions; a++) {
    //         v[s] += actionProbabilities[s](a) * Q[1](s, a);
    //     }
    //     printf("V(%d): %f\n", s, v[s]);
    // }

    // IGNORE below, used for policy eval from start states
    evalPolicyValue = 0.0;
    for (int s=0; s < numStates; s++) {
        for (int a=0; a < numActions; a++) {
            evalPolicyValue += d0[s] * actionProbabilities[s](a) * Q[0](s, a);
        }
    }
}

