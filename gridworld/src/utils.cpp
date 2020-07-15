#include <map>
#include "utils.h"
#include "Trajectory.h"
#include "Environment.hpp"
#include "StochasticGridworldNonTrajLim.h"
#include <set>

int get_visited_s_a_s_a_count(vector<Trajectory> &all_data, int num_states, int num_actions) {


    std::set<std::tuple<int,int,int,int>> counts;
    std::set<std::tuple<int,int,int,int>>::iterator it;

    int count = 0;
    for (auto &traj : all_data) {
        for (int t = 0; t < traj.len; t++) {
            int s = traj.states[t];
            int a = traj.actions[t];
            int sPrime = -1;
            int aPrime = -1;
            if (t < traj.len - 1){
                sPrime = traj.states[t + 1];
                aPrime = traj.actions[t + 1];
            }
            std::tuple<int, int, int,int> tu = std::make_tuple(s, a, sPrime, aPrime);
            it = counts.find(tu);
            if (it == counts.end()){
                counts.insert(tu);
                count++;
            }
        }
    }
    return count;
}

int get_visited_s_a_count(vector<Trajectory> &all_data, int num_states, int num_actions) {

    MatrixXd s_a_counts = MatrixXd::Zero(num_states, num_actions);

    int count = 0;
    for (auto &traj : all_data) {
        for (int t = 0; t < traj.len; t++) {
            int s = traj.states[t];
            int a = traj.actions[t];
            if (s_a_counts(s, a) == 0) count++;
            s_a_counts(s, a) += 1;
        }
    }
    return count;
}

void LoadTrajIWs(vector<Trajectory> & trajs, const Policy &eval_pi) {
  for (int i = 0; i < static_cast<int>(trajs.size()); i++) {
    trajs[i].IWs.resize(trajs[i].len);
    trajs[i].cumIWs.resize(trajs[i].len);
    trajs[i].evalActionProbabilities.resize(trajs[i].len);
    for (int t = 0; t < trajs[i].len; t++) {
      trajs[i].evalActionProbabilities[t] = eval_pi.getActionProbability(
        trajs[i].states[t], trajs[i].actions[t]);
      trajs[i].IWs[t] = trajs[i].evalActionProbabilities[t] /
        trajs[i].actionProbabilities[t];
      trajs[i].cumIWs[t] = trajs[i].IWs[t];
      if (t != 0)
        trajs[i].cumIWs[t] *= trajs[i].cumIWs[t-1];
    }
  }
}


MatrixXd getIWs(const vector<Trajectory> trajs, const bool & weighted, const int & L) {
  int N = (int)trajs.size();
  MatrixXd rhot(N, L);
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < L; t++) {
      if (t < trajs[i].len)
        rhot(i,t) = trajs[i].cumIWs[t];
      else
        rhot(i,t) = trajs[i].cumIWs[trajs[i].len-1];
    }
  }
  if (weighted) {
    for (int t = 0; t < L; t++)
      rhot.col(t) = rhot.col(t) / rhot.col(t).sum();
  }
  else
    rhot = rhot / (double)N;
  return rhot;
}


double getISVariance(vector<Trajectory> data, double control_variate) {
  double estimate = 0.0;
  double square_estimate = 0.0;
  double val;
  for (auto & traj : data) {
    val = traj.cumIWs[traj.len - 1] * (traj.R - control_variate);
    estimate += val;
    square_estimate += pow(val + control_variate, 2);
  }
  double mean = (estimate / data.size()) + control_variate;
  double uncentered_variance = square_estimate / data.size();
  return uncentered_variance - (mean * mean);
}

Environment* getDomain(int domain, float deterministic_prob) {
  if (domain == STOCHASTIC_GRIDWORLD_NON_TRAJ_LIM) {
    printf("Stochastic Gridworld no traj lim\n");
    return new StochasticGridworldNonTrajLim(false, deterministic_prob);
  } else {
    printf("Unknow domain number...exiting\n");
    exit(1);
  }
}

string get_method_name(int method) {
  if (method == 0) {
    printf("TD\n");
    return "TD";
  } else if (method == 1) {
    printf("RIS TD\n");
    return "RisTD";
  } else {
    printf("Unknow method number...exiting\n");
    exit(1);
  }
}

void MLE_LoadTrajIWs(vector<Trajectory> &estimate_data, vector<Trajectory> &pib_data,
                     const Policy &eval_pi, const Environment &env,
                     int smoothing) {
  // Get counts for each (s,a) pair
  int k = smoothing;  // Laplace smoothing parameter
  int L = env.getMaxTrajLen();
  int numStates = env.getNumStates();
  int numActions = env.getNumActions();
  VectorXd state_counts = VectorXd::Zero(env.getNumStates());
  MatrixXd probs = MatrixXd::Zero(env.getNumStates(), env.getNumActions());

  for (int s=0; s < numStates; s++) {
    state_counts(s) += k * numActions;
    for (int a=0; a < numActions; a++) {
      probs(s, a) += k;
    }
  }
  for (auto & traj : pib_data) {
    for (int t=0; t < traj.len; t++) {
      probs(traj.states[t], traj.actions[t]) += 1;
      state_counts(traj.states[t]) += 1;
    }
  }
  // for (int t=0; t < L; t++){
  for (int s = 0; s < numStates; s++) {
    for (int a = 0; a < numActions; a++) {
      if (state_counts(s) > 0)
        probs(s, a) /= state_counts(s);
    }
  }

  double action_prob;
  int state, action;
  for (auto & traj : estimate_data) {
    traj.IWs.resize(traj.len);
    traj.cumIWs.resize(traj.len);
    traj.evalActionProbabilities.resize(traj.len);
    for (int t = 0; t < traj.len; t++) {
      state = traj.states[t];
      action = traj.actions[t];
      action_prob = eval_pi.getActionProbability(state, action);
      traj.evalActionProbabilities[t] = action_prob;
      action_prob = probs(state, action);
      // action_prob = state_action_time_counts[t][state][action];
      traj.IWs[t] = traj.evalActionProbabilities[t] / action_prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
    }
  }
}

void RISN_LoadTrajIWs(std::vector<Trajectory> &data, const Policy &eval_pi, int n,
                      bool reset) {
  static std::map<std::vector<int>, int> counts;
  static std::map<std::vector<int>, int> state_counts;
  static int start = 0;
  if (reset) {
    counts.clear();
    state_counts.clear();
    start = 0;
  }
  std::map<std::vector<int>, double> probs;
  std::vector<int> state_seg;
  std::vector<int> action_seg;
  std::map<std::vector<int>, int>::iterator it;
  for (int i=start; i < data.size(); i++) {
    state_seg.clear();
    action_seg.clear();
    for (int t=0; t < data[i].len; t++) {
      state_seg.push_back(data[i].states[t]);
      action_seg.push_back(data[i].states[t]);
      action_seg.push_back(data[i].actions[t]);
      it = counts.find(action_seg);
      if (it == counts.end()) {
        state_counts.insert(std::pair<vector<int>, int>(state_seg, 0));
        counts.insert(std::pair<vector<int>, int>(action_seg, 0));
        probs.insert(std::pair<vector<int>, double>(action_seg, 0.0));
      }
      state_counts[state_seg] += 1;
      counts[action_seg] += 1;
      state_seg.push_back(data[i].actions[t]);

      if (action_seg.size() >= 2 * n) {
        state_seg.erase(state_seg.begin(), state_seg.begin() + 2);
        action_seg.erase(action_seg.begin(), action_seg.begin() + 2);
      }
    }
  }
  start = data.size();
  it = state_counts.begin();
  int len = 0;
  for (map<vector<int>, int>::iterator it = counts.begin();
       it != counts.end(); ++it) {
    std::vector<int> new_state_seg(it->first);
    new_state_seg.pop_back();
    probs[it -> first] = static_cast<double>(it -> second) /
      state_counts[new_state_seg];
  }

  double action_prob;
  int state, action;
  for (auto & traj : data) {
    state_seg.clear();
    action_seg.clear();
    traj.IWs.resize(traj.len);
    traj.cumIWs.resize(traj.len);
    traj.evalActionProbabilities.resize(traj.len);
    for (int t = 0; t < traj.len; t++) {
      state_seg.push_back(traj.states[t]);
      action_seg.push_back(traj.states[t]);
      action_seg.push_back(traj.actions[t]);
      state = traj.states[t];
      action = traj.actions[t];
      action_prob = eval_pi.getActionProbability(state, action);
      traj.evalActionProbabilities[t] = action_prob;
      action_prob = probs[action_seg];
      traj.IWs[t] = traj.evalActionProbabilities[t] / action_prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
      state_seg.push_back(traj.actions[t]);
      if (action_seg.size() >= 2*n) {
        state_seg.erase(state_seg.begin(), state_seg.begin() + 2);
        action_seg.erase(action_seg.begin(), action_seg.begin() + 2);
      }
    }
  }
}


void saveHeatMap(string outfile, int numStates, int numActions,
                 vector<Trajectory> & data) {
  int** counts = new int*[numStates];
  for (int i=0; i < numStates; i++) {
    counts[i] = new int[numActions];
    for (int j=0; j < numActions; j++)
      counts[i][j] = 0;
  }
  for (auto & traj : data) {
    for (int t=0; t < traj.len; t++)
      counts[traj.states[t]][traj.actions[t]]++;
  }
  ofstream out;
  out.open(outfile);
  for (int i=0; i < numStates; i++) {
    for(int j=0; j < numActions; j++) {
      if (j > 0) out << " ";
      out << counts[i][j];
    }
    out << endl;
  }
  out.close();
}

void saveReturns(string outfile, vector<Trajectory> & data) {
  ofstream out;
  out.open(outfile);
  for (auto & traj : data) out << traj.R << endl;
  out.close();
}
