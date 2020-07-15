#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include "Trajectory.h"
#include <vector>
#include "Policy.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
This class builds a model of an MDPs with tabular states and actions, and a finite  horizon, L.
*/

class Model {
public:
	// Adding a defauly constructor so that I can extend it
	Model();

	/*
	Take historical data in trajs and build a model. numState and numActions are the total numbers of possible states and actions
	*/
	Model(const vector<Trajectory*> & trajs, int numStates, int numActions, int L, bool JiangStyle);

	/*
	Use the model to directly estimate the value of a policy. This uses value iteration.
	*/
	//double getPolicyValue(const Policy & pi) const;

	// V and Q predictions - can be loaded for a specific evaluation policy
	void loadEvalPolicy(const Policy & pi, const int & L); // int L = Gridworld::getMaxTrajLen();
	vector<VectorXd> actionProbabilities; // [s][a]
	vector<VectorXd> V; // [t](s) - t in [0,L].
	vector<MatrixXd> Q; // [t](s,a)
	//vector<MatrixXd> Rsa; // [t](s,a) - Prediction of R_t given that S_0 =s and A_0=a
	//vector<vector<MatrixXd>> Rsas; // [t][s](a,sPrime) - Prediction of R_t given that S_0 =s and A_0=a, and S_1 = sPrime
	//MatrixXd Rs; // (s,t)
	double evalPolicyValue;

	// Generate trajectories from the provided policy
	vector<Trajectory> generateTrajectories(const Policy & pi, int N, mt19937_64 & generator) const;

	// Estimate value of policy under model using Monte Carlo returns
    double evalMonteCarlo(const Policy & pi, int N, mt19937_64 & generator) const;

	// We exposed these variables in order to do DRv2 (version 2 implementations) without having to compute Rsa and Rsas tables above (compute intensive)
	vector<vector<vector<double>>> R;				// R[s][a][s']. Size = [numStates][numActions][numStates+1]
	vector<vector<vector<double>>> P;				// P[s][a][s']. Size = [numStates][numActions][numStates+1]

    // certainty equivalence related
    // TODO: make more general once RIS version done
    vector<VectorXd> V_cee_mrp;
    void load_DP_CEE_MRP(const int & L, vector<Trajectory> &traj);
    void compute_CEE_MRP(const int & L, vector<Trajectory> &traj);
    vector<int> stateCounts;
    vector<vector<int>> stateStateCounts;
    vector<vector<double>> stateStateRewardSum;

    vector<VectorXd> s_counts;
    vector<MatrixXd> s_s_counts;
    vector<MatrixXd> s_s_r_sums;


    // CEE for RIS
    vector<MatrixXd> Q_cee_mdp;
    vector<VectorXd> V_cee_mdp;
    void load_DP_CEE_MDP(const Policy &pi,const int & L, vector<Trajectory> &traj);
    void load_Q_DP(const Policy & pi, const int & L, vector<Trajectory> &all_data, string dp_type);
    void compute_CEE_MDP(const int & L, vector<Trajectory> &traj);
    vector<MatrixXd> s_a_counts;
    vector<vector<MatrixXd>> s_a_s_counts;
    vector<vector<vector<MatrixXd>>> s_a_s_a_counts;
    vector<vector<MatrixXd>> s_a_s_r_sums;


//private:
	int N;
	int L;
	int numStates;
	int numActions;
	vector<double> d0;								// d0[s] = Pr(S_0=s). Size = [numStates]

	// How many times was each (s), (s,a), and (s,a,s') tuple seen?
	vector<vector<int>> stateActionCounts;
	vector<vector<int>> stateActionCounts_includingHorizon;				 // Includes transitions to terminal absorbing state due to time horizon
	vector<vector<vector<int>>> stateActionStateCounts;
	vector<vector<vector<int>>> stateActionStateCounts_includingHorizon; // Includes transitions to terminal absorbing state due to time horizon

    const double eps = 1e-5;//1e-10;
};

#endif
