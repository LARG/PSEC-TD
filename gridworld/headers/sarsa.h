#ifndef _SARSA_HPP_
#define _SARSA_HPP_

#include "Trajectory.h"
#include <vector>
#include "Policy.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
This class estimates the value of a policy with expected SARSA
*/

class Sarsa {
public:

    // Adding a defauly constructor so that I can extend it
    Sarsa();

    /*
    Take historical data in trajs and learns a value function
    */
    Sarsa(int numStates, int numActions, int L);

    void loadEvalPolicy(const vector<Trajectory*> &trajs, Policy evalPolicy);

    vector<VectorXd> actionProbabilities; // [s][a]
    vector<MatrixXd> Q; // [s][a]
    double evalPolicyValue;

    int L;
    int numStates;
    int numActions;
    vector<double> d0;              // d0[s] = Pr(S_0=s). Size = [numStates]

};

#endif
