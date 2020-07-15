#include <vector>

#include "Trajectory.h"
#include "Environment.hpp"

#define MOUNTAINCAR 0
#define MODELWIN 1
#define MODELFAIL 2
#define SMALL_GRID_TH 3
#define GRIDWORLD_TH 4
#define GRIDWORLD_FH 5
#define SIMPLE 6
#define BIG_GRID 7
#define SMALL_GRID_FH 8
#define STOCHASTIC_GRIDWORLD 9
#define RARE_EVENT_GW 10
#define STOCHASTIC_GRIDWORLD_FH 11
#define GRIDWORLD_NON_TRAJ_LIM 12
#define CHAIN_NON_TRAJ_LIM 13
#define STOCHASTIC_GRIDWORLD_NON_TRAJ_LIM 14

void LoadTrajIWs(vector<Trajectory> & trajs, const Policy &eval_pi);

int get_visited_s_a_s_a_count(vector<Trajectory> &all_data, int num_states, int num_actions); 
int get_visited_s_a_count(vector<Trajectory> & trajs, int num_states, int num_actions);

MatrixXd getIWs(const vector<Trajectory> trajs, const bool & weighted, const int & L);

double getISVariance(vector<Trajectory> data, double control_variate) ;


Environment* getDomain(int domain, float deterministic_prob);

string get_method_name(int domain);

void MLE_LoadTrajIWs(vector<Trajectory> &estimate_data,
										 vector<Trajectory> &pib_data, const Policy &eval_pi,
                     const Environment &env, int smoothing);

void RISN_LoadTrajIWs(vector<Trajectory> &data, const Policy &eval_pi, int n,
                      bool reset);


void saveHeatMap(string outfile, int numStates, int numActions,
                 vector<Trajectory> & data);

void saveReturns(string outfile, vector<Trajectory> & data);
