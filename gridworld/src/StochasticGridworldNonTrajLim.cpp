#include "StochasticGridworldNonTrajLim.h"

const int g_GRIDWORLD_SIZE = 4;
//const int g_GRIDWORLD_MAX_TRAJLEN = 50;
const int BIG_REWARD = 100;
const int BIG_PENALTY = -10;
//const float MAX_PROB = 0.5;

StochasticGridworldNonTrajLim::StochasticGridworldNonTrajLim(bool trueHorizon, float deterministic_prob) {
  this->trueHorizon = trueHorizon;
  this->MAX_PROB = deterministic_prob;
}

int StochasticGridworldNonTrajLim::getNumActions() const {
  return 4;
}

int StochasticGridworldNonTrajLim::getNumStates() const {
  return g_GRIDWORLD_SIZE * g_GRIDWORLD_SIZE - 1;
}

int StochasticGridworldNonTrajLim::getMaxTrajLen() const {
    return 1;
}

double StochasticGridworldNonTrajLim::getMinReturn() {return -250;}
double StochasticGridworldNonTrajLim::getMaxReturn() {return 50;}

int StochasticGridworldNonTrajLim::getNumEvalTrajectories() {
  return 10000;
}

int StochasticGridworldNonTrajLim::getNextState(int state, int action,
                                      mt19937_64 & generator) {
  VectorXd probs(3);
  probs(0) = MAX_PROB;
  probs(1) = 0.5 - 0.5 * MAX_PROB;
  probs(2) = 0.5 - 0.5 * MAX_PROB;
  int x = state % g_GRIDWORLD_SIZE;
  int y = state / g_GRIDWORLD_SIZE;
  int rand = wrand(generator, probs);
  if (rand == 0) {  // max likelihood transition
    if ((action == 0) && (x < g_GRIDWORLD_SIZE - 1))  // right
      x++;
    else if ((action == 1) && (x > 0))  // left
      x--;
    else if ((action == 2) && (y < g_GRIDWORLD_SIZE - 1))  // down
      y++;
    else if ((action == 3) && (y > 0))  // up
      y--;
  } else if (rand == 1) {  // right of intention
    if ((action == 0) && (y < g_GRIDWORLD_SIZE - 1))
      y++;
    else if ((action == 1) && (y > 0))
      y--;
    else if ((action == 2) && (x > 0 ))
      x--;
    else if ((action == 3) && (x < g_GRIDWORLD_SIZE - 1))
      x++;
  } else if (rand == 2) {  // left of intention
    if ((action == 0) && (y > 0))
      y--;
    else if ((action == 1) && (y < g_GRIDWORLD_SIZE - 1))
      y++;
    else if ((action == 2) && (x < g_GRIDWORLD_SIZE - 1))
      x++;
    else if ((action == 3) && (x > 0))
      x--;
  }
  return x + y*g_GRIDWORLD_SIZE;
}

void StochasticGridworldNonTrajLim::generateTrajectories(vector<Trajectory> & buff,
                                               const Policy & pi, int numTraj,
                                               mt19937_64 & generator) {
  buff.resize(numTraj);
  for (int trajCount = 0; trajCount < numTraj; trajCount++) {
    buff[trajCount].len = 0;
    buff[trajCount].actionProbabilities.resize(0);
    buff[trajCount].actions.resize(0);
    buff[trajCount].rewards.resize(0);
    buff[trajCount].states.resize(1);
    buff[trajCount].R = 0;
    int x = 0, y = 0;
    int t = 0;
    buff[trajCount].states[0] = x + y*g_GRIDWORLD_SIZE;
    while (true) {
      buff[trajCount].len++;  // We have one more transition!
      // Get action
      int action = pi.getAction(buff[trajCount].states[t], generator);
      buff[trajCount].actions.push_back(action);
      double actionProbability = pi.getActionProbability(
        buff[trajCount].states[t], buff[trajCount].actions[t]);
      buff[trajCount].actionProbabilities.push_back(actionProbability);

      // Get next state and reward
      int next = getNextState(x +y*g_GRIDWORLD_SIZE, action, generator);
      x = next % g_GRIDWORLD_SIZE;
      y = next / g_GRIDWORLD_SIZE;

      // Update the reward
      double reward;
      if ((x == 1) && (y == 1))
        reward = BIG_PENALTY;
      else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1))
        reward = 1;
      else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))
        reward = BIG_REWARD; //5 * BIG_REWARD;
      else
        reward = -1;
      buff[trajCount].rewards.push_back(reward);
      buff[trajCount].R += reward;

      if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        // Entered a terminal state. Last transition
        break;
      }

      // Add the state and features for the next element
      buff[trajCount].states.push_back(x + y*g_GRIDWORLD_SIZE);
      t++;
    }
  }
}

double StochasticGridworldNonTrajLim::evaluatePolicy(const Policy & pi,
                                           mt19937_64 & generator) {
  int numSamples = 10000;

  double result = 0;
  for (int trajCount = 0; trajCount < numSamples; trajCount++) {
    int x = 0, y = 0;
    while (true) {
      int action = pi.getAction(x + y*g_GRIDWORLD_SIZE, generator);
      int next = getNextState(x +y*g_GRIDWORLD_SIZE, action, generator);
      x = next % g_GRIDWORLD_SIZE;
      y = next / g_GRIDWORLD_SIZE;

      // Update reward
      if ((x == 1) && (y == 1))
        result += BIG_PENALTY;
      else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1))
        result += 1;
      else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))
        result += BIG_REWARD;//5*BIG_REWARD;
      else
        result += -1;

      if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        // Entered a terminal state. Last transition
        break;
      }
    }
  }
  return result / static_cast<double>(numSamples);
}

Policy StochasticGridworldNonTrajLim::getPolicy(int index) {
  if (index == 1)
    return Policy("policies/stochastic_gridworld/p1.txt",
      getNumActions(), getNumStates());
  if (index == 2)
    return Policy("policies/stochastic_gridworld/p2.txt",
      getNumActions(), getNumStates());
  if (index == 3)
    return Policy("policies/stochastic_gridworld/p3.txt",
      getNumActions(), getNumStates());
  if (index == 4)
    return Policy("policies/stochastic_gridworld/p4.txt",
      getNumActions(), getNumStates());
  if (index == 5)
    return Policy("policies/stochastic_gridworld/p5.txt",
      getNumActions(), getNumStates());
  if (index % 500 == 0) {
    ostringstream ss;
    ss << "policies/stochastic_gridworld/stochastic_gridworld_p";
    ss << index << ".txt";
    return Policy(ss.str().c_str(), getNumActions(), getNumStates());
  }
  errorExit("Error091834");
  return Policy("error", 0, 0);
}

Model StochasticGridworldNonTrajLim::getTrueModel() {
  int numStates = g_GRIDWORLD_SIZE * g_GRIDWORLD_SIZE - 1;
  int numActions = 4;
  int dim = g_GRIDWORLD_SIZE;
  vector<Trajectory*> trajs;
  Model model(trajs, numStates, numActions, 1, false);
  for (int i=0; i < numStates; i++) model.d0[i] = 0;
  model.d0[0] = 1.0;
  printf("%d %d %d %d\n", numStates, numActions, 1, dim);
  for (int s=0; s < numStates; s++) {
    for (int a=0; a < numActions; a++) {
      for (int j=0; j <  numStates + 1; j++) model.P[s][a][j] = 0.0;
      for (int j=0; j <  numStates + 1; j++) model.R[s][a][j] = 0.0;
      for (int rand =0; rand < 3; rand++) {
        int x = s % dim; int y = s / dim; double rew = -1;
        int sPrime;
        if (rand == 0) {  // max likelihood transition
          if ((a == 0) && (x < g_GRIDWORLD_SIZE - 1))  // right
            x++;
          else if ((a == 1) && (x > 0))  // left
            x--;
          else if ((a == 2) && (y < g_GRIDWORLD_SIZE - 1))  // down
            y++;
          else if ((a == 3) && (y > 0))  // up
            y--;
          sPrime = x + y * dim;
          model.P[s][a][sPrime] = MAX_PROB;
        } else if (rand == 1) {  // right of intention
          if ((a == 0) && (y < g_GRIDWORLD_SIZE - 1))
            y++;
          else if ((a == 1) && (y > 0))
            y--;
          else if ((a == 2) && (x > 0 ))
            x--;
          else if ((a == 3) && (x < g_GRIDWORLD_SIZE - 1))
            x++;
          sPrime = x + y * dim;
          model.P[s][a][sPrime] += 0.5 - MAX_PROB*0.5;
        } else if (rand == 2) {  // left of intention
          if ((a == 0) && (y > 0))
            y--;
          else if ((a == 1) && (y < g_GRIDWORLD_SIZE - 1))
            y++;
          else if ((a == 2) && (x < g_GRIDWORLD_SIZE - 1))
            x++;
          else if ((a == 3) && (x > 0))
            x--;
          sPrime = x + y * dim;
          model.P[s][a][sPrime] += 0.5 - MAX_PROB*0.5;
        }
      if ((x == 1) && (y == 1))
        model.R[s][a][sPrime] = BIG_PENALTY;
      else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1))
        model.R[s][a][sPrime] = 1;
      else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))
        model.R[s][a][sPrime] = BIG_REWARD;
      else
        model.R[s][a][sPrime] = -1;
      }
    }
  }
  return model;
}

double StochasticGridworldNonTrajLim::getTrueValue(int policy_number) {
  Model m = getTrueModel();
  Policy policy = getPolicy(policy_number);
  int horizon = 1;
  m.loadEvalPolicy(policy, horizon);
  return m.evalPolicyValue;
}

double StochasticGridworldNonTrajLim::getTrueValue(const Policy & pi) {
  Model m = getTrueModel();
  int horizon = 1;
  m.loadEvalPolicy(pi, horizon);
  return m.evalPolicyValue;
}
