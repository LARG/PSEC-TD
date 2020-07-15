#include "Model.hpp"
Model::Model() {
}

Model::Model(const vector<Trajectory*> & trajs, int numStates, int numActions, int L, bool JiangStyle) {
	
    if (L <= 0) {
       std::cout << "Max traj len invalid" << std::endl;
       exit(-1);
    }

    // Get the number of states and actions that have actually been observed
	this->numStates = numStates;
	this->numActions = numActions;
	this->L = L;

	N = (int)trajs.size();

//	for (int j=0; j < trajs.size(); j++)
//		cout << "In Model Length, " << trajs[j]->len << endl;

    // certainty equivalence
    std::cout << "setting state counts... " << std::endl;
    stateCounts.resize(numStates + 1);
    stateStateCounts.resize(numStates + 1);
    stateStateRewardSum.resize(numStates + 1);
    
	// Resize everything and set all to zero
	stateActionCounts.resize(numStates);
	stateActionCounts_includingHorizon.resize(numStates);
	stateActionStateCounts.resize(numStates);
	stateActionStateCounts_includingHorizon.resize(numStates);
	P.resize(numStates);
	R.resize(numStates);
	d0.resize(numStates);
    std::cout << "number of states " << numStates << std::endl;
	for (int s = 0; s < numStates; s++) {
        // certainty equiv related
        std::cout << "setting statestate counts... " << std::endl;
        stateStateCounts[s].resize(numStates + 1);
        stateStateRewardSum[s].resize(numStates + 1);

		stateActionCounts[s].resize(numActions);
		stateActionCounts_includingHorizon[s].resize(numActions);
		stateActionStateCounts[s].resize(numActions);
		stateActionStateCounts_includingHorizon[s].resize(numActions);
		P[s].resize(numActions);
		R[s].resize(numActions);
		d0[s] = 0;
        stateCounts[s] = 0;
        stateCounts[s + 1] = 0;
		for (int a = 0; a < numActions; a++) {
			stateActionCounts[s][a] = 0;
			stateActionCounts_includingHorizon[s][a] = 0;
			stateActionStateCounts[s][a].resize(numStates + 1);
			stateActionStateCounts_includingHorizon[s][a].resize(numStates + 1);
			P[s][a].resize(numStates + 1);
			R[s][a].resize(numStates + 1);
			for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
				stateActionStateCounts[s][a][sPrime] = 0;
				stateActionStateCounts_includingHorizon[s][a][sPrime] = 0;
				P[s][a][sPrime] = 0;
				R[s][a][sPrime] = 0;
                stateStateCounts[s][sPrime] = 0;
                stateStateRewardSum[s][sPrime] = 0;
			}
		}
	}
	if (N > 0) {
		// Compute all of the counts, and set R to the sum of rewards from the [s][a][sPrime] transitions
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < trajs[i]->len; j++) {
	//			cout << "Lengths: " << trajs[i]->len << endl;
				int s = trajs[i]->states[j], a = trajs[i]->actions[j], sPrime = (j == trajs[i]->len - 1 ? numStates : trajs[i]->states[j + 1]);
				double r = trajs[i]->rewards[j];

                int end_len = L - 1;
                if (L == 1) {
                    // TODO verify this is correct
                    end_len = trajs[i]->len;
                }

				if (j != end_len) {
					stateActionCounts[s][a]++;
					stateActionStateCounts[s][a][sPrime]++;
				}
				else {
					// Tested, and we do get here.
					if (sPrime != numStates)
						errorExit("argh234523456");
				}
				stateActionCounts_includingHorizon[s][a]++;
				stateActionStateCounts_includingHorizon[s][a][sPrime]++;

				R[s][a][sPrime] += r;
			}
		}

		// Compute d0
		for (int i = 0; i < N; i++)
			d0[trajs[i]->states[0]] += 1.0 / (double)N;

		// Compute rMin - used by Nan Jiang's model style from Doubly Robust paper
		double rMin = trajs[0]->rewards[0];
		for (int i = 0; i < N; i++) {
			for (int t = 0; t < trajs[i]->len; t++)
				rMin = min(trajs[i]->rewards[t], rMin);
		}

		// Compute P and R
		for (int s = 0; s < numStates; s++) {
			for (int a = 0; a < numActions; a++) {
				for (int sPrime = 0; sPrime < numStates+1; sPrime++) {
					if (stateActionCounts[s][a] == 0) {
						if (JiangStyle)
							P[s][a][sPrime] = (sPrime == s ? 1 : 0); // Self transition
						else
							P[s][a][sPrime] = (sPrime == numStates ? 1 : 0); // Assume termination
					}
					else
						P[s][a][sPrime] = (double)stateActionStateCounts[s][a][sPrime] / (double)stateActionCounts[s][a];

					if (stateActionStateCounts_includingHorizon[s][a][sPrime] == 0) {
						if (JiangStyle)
							R[s][a][sPrime] = rMin;
						else
							R[s][a][sPrime] = 0;							// No data - don't divide by zero
					}
					else
						R[s][a][sPrime] /= (double)stateActionStateCounts_includingHorizon[s][a][sPrime];
				}
			}
		}
	}
//	for (int i=0 ; i < d0.size(); i++)
//          if (d0[i] > 0)
//		cout << i << " " << d0[i] << endl;
//
        std::cout << "exiting constructor " << std::endl;
}

void Model::load_DP_CEE_MRP(const int & L, vector<Trajectory> & all_data) {
	
    compute_CEE_MRP(L, all_data);
    bool print =true;
    /*
    for(int i=0; i<stateStateCounts.size(); ++i)
        for (int j = 0; j < stateStateCounts[i].size(); ++j)
            std::cout << stateStateCounts[i][j] << ' ';
    */

    for(int i=0; i<stateCounts.size(); ++i)
      std::cout << stateCounts[i] << ' ';

	V_cee_mrp.resize(L);
    vector<VectorXd> temp_V;
    temp_V.resize(L);
     
	for (int t = L-1; t >= 0; t--) {
        V_cee_mrp[t] = VectorXd::Zero(numStates + 1);
        temp_V[t] = VectorXd::Zero(numStates + 1);
        int lim = 0;
        while (true) {
        int num = 0;
		for (int s = 0; s < numStates; s++) {
            
            double res = 0;
            for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
                if (s_counts[t](s) == 0 /*stateCounts[s] == 0*/) {
                    // if never visited state, no expected return anyway
                    continue;
                }
                if (s_s_counts[t](s, sPrime) == 0 /*stateStateCounts[s][sPrime] == 0*/) {
                    // if never visited sPrime from s, then skip this sPrime
                    continue;
                }
                
                double trans_est = double(s_s_counts[t](s, sPrime)) / s_counts[t](s); //stateCounts[s];
                double reward_est = double(s_s_r_sums[t](s,sPrime)) / s_s_counts[t](s, sPrime); //stateStateCounts[s][sPrime];

                //double trans_est = double(stateStateCounts[s][sPrime]) / stateCounts[s];
                //double reward_est = double(stateStateRewardSum[s][sPrime]) / stateStateCounts[s][sPrime];
                
                //V_cee_mrp[t](s) += trans_est * reward_est;
                res += trans_est * reward_est;
                if (L != 1) {
                    if ((sPrime != numStates) && (t != L-1))
                        V_cee_mrp[t](s) += trans_est * V_cee_mrp[t + 1](sPrime);
                } else {
                    if ((sPrime != numStates))
                        // t = 0 always in this case
                        //V_cee_mrp[t](s) += trans_est * V_cee_mrp[t](sPrime);
                        res += trans_est * V_cee_mrp[t](sPrime);
                }
            }

            temp_V[t](s) = V_cee_mrp[t](s);
            V_cee_mrp[t](s) = res;
            double diff = abs(V_cee_mrp[t](s) - temp_V[t](s));
            //std::cout << diff << std::endl;
            if (diff <= eps) num++;
		}
        if (num == numStates) {
            std::cout << "CEE MRP converged" << std::endl;
            break;
        }
        lim++;
        }
	}
    /*
	for (int t = L-1; t >= 0; t--) {
        V_cee_mrp[t] = VectorXd::Zero(numStates + 1);
		for (int s = 0; s < numStates; s++) {
            if (s_counts[t](s) == 0) {
                // if never visited state, no expected return anyway
                continue;
            }
            for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
                if (s_s_counts[t](s, sPrime) == 0) {
                    // if never visited sPrime from s, then skip this sPrime
                    continue;
                }
                
                double trans_est = double(s_s_counts[t](s, sPrime)) / s_counts[t](s); //stateCounts[s];
                double reward_est = double(s_s_r_sums[t](s,sPrime)) / s_s_counts[t](s, sPrime); //stateStateCounts[s][sPrime];

                //double trans_est = double(stateStateCounts[s][sPrime]) / stateCounts[s];
                //double reward_est = double(stateStateRewardSum[s][sPrime]) / stateStateCounts[s][sPrime];
                V_cee_mrp[t](s) += trans_est * reward_est;
                if (L != 1) {
                    if ((sPrime != numStates) && (t != L-1))
                        V_cee_mrp[t](s) += trans_est * V_cee_mrp[t + 1](sPrime);
                } else {
                    if ((sPrime != numStates))
                        // t = 0 always in this case
                        V_cee_mrp[t](s) += trans_est * V_cee_mrp[t](sPrime);
                }
            }
		}
	}
    */
}

void Model::compute_CEE_MRP(const int &L, vector<Trajectory> &all_data) {

    s_counts.resize(L);
    s_s_counts.resize(L);
    s_s_r_sums.resize(L);

    for (int i = 0; i < L; i++){
        s_counts[i] = VectorXd::Zero(numStates + 1);
        s_s_counts[i] = MatrixXd::Zero(numStates + 1, numStates + 1);
        s_s_r_sums[i] = MatrixXd::Zero(numStates + 1, numStates + 1);
    }
    int s, a, sPrime;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        sPrime = (t == traj.len - 1) ? numStates : traj.states[t + 1];
        stateCounts[s] += 1;
        stateStateCounts[s][sPrime] += 1;
        stateStateRewardSum[s][sPrime] += traj.rewards[t];

        int time_index = 0;
        if (L != 1) {
            time_index = t;
        }
        s_counts[time_index](s) += 1;
        s_s_counts[time_index](s, sPrime) += 1;
        s_s_r_sums[time_index](s, sPrime) += traj.rewards[t]; 
      }
    } 
}

void Model::load_DP_CEE_MDP(const Policy & pi, const int & L, vector<Trajectory> &all_data) {
	//cout << "Starting..." << endl;
	// Load actionProbabilities - so we only compute them once
	actionProbabilities.resize(numStates);
	for (int s = 0; s < numStates; s++){
		actionProbabilities[s] = pi.getActionProbabilities(s);
		// cout << s << " " << actionProbabilities[s] << endl;
	}

    compute_CEE_MDP(L, all_data);
	// Q[t](s,a) = Q(s,a) given that S_t=s, A_t=a, and at t=L the state is absorbing.
	
	Q_cee_mdp.resize(L);
    vector<VectorXd> temp_V;
    temp_V.resize(L);
	V_cee_mdp.resize(L);
	
    for (int t = L-1; t >= 0; t--) {
		Q_cee_mdp[t] = MatrixXd::Zero(numStates+1, numActions);
        temp_V[t] = VectorXd::Zero(numStates + 1);
		V_cee_mdp[t] = VectorXd::Zero(numStates + 1);

        while (true) {
            //cout << "Time step " << t << endl;
            for (int s = 0; s < numStates; s++) {
                for (int a = 0; a < numActions; a++) {
                    double res = 0;
                    if (s_a_counts[t](s, a) == 0) {
                        continue;
                    }
                    for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
                        if (s_a_s_counts[t][s](a, sPrime) == 0) {
                            continue;
                        }
                        double trans_est = double(s_a_s_counts[t][s](a, sPrime)) / s_a_counts[t](s, a);
                        double reward_est = double(s_a_s_r_sums[t][s](a, sPrime)) / s_a_s_counts[t][s](a, sPrime); 
                        //double reward_est = R[s][a][sPrime];
                        res += trans_est * reward_est;
                        
                        //Q_cee_mdp[t](s,a) += trans_est * reward_est;
                        if (L != 1) {
                            if ((sPrime != numStates) && (t != L-1))
                                Q_cee_mdp[t](s, a) += trans_est * actionProbabilities[sPrime].dot(Q_cee_mdp[t+1].row(sPrime));
                        } else {
                            if ((sPrime != numStates))
                                // t = 0 always in this case
                                res += trans_est * actionProbabilities[sPrime].dot(Q_cee_mdp[t].row(sPrime)); 
                                //Q_cee_mdp[t](s, a) += trans_est * actionProbabilities[sPrime].dot(Q_cee_mdp[t].row(sPrime));
                        }
                    }
                    Q_cee_mdp[t](s, a) = res;
                }
            }
            int num = 0;
            temp_V[t] = V_cee_mdp[t];
            // Load V[t](s) = V(s) given that S_t = s and at t=L the state is absorbing.
            for (int s = 0; s < numStates; s++) {
                V_cee_mdp[t][s] = actionProbabilities[s].dot(Q_cee_mdp[t].row(s));
                if (abs(temp_V[t](s) - V_cee_mdp[t](s)) <= eps) num++;
            }
            if (num == numStates) {
                std::cout << "CEE MDP converged" << std::endl;
                break;
            }
        }
	}
}

void Model::compute_CEE_MDP(const int &L, vector<Trajectory> &all_data) {

    s_counts.resize(L);
    s_a_counts.resize(L);
    s_a_s_counts.resize(L);
    s_a_s_r_sums.resize(L);
    s_a_s_a_counts.resize(L);

    for (int i = 0; i < L; i++){
        s_counts[i] = VectorXd::Zero(numStates + 1);
        s_a_counts[i] = MatrixXd::Zero(numStates + 1, numActions);
        s_a_s_counts[i].resize(numStates + 1);
        s_a_s_r_sums[i].resize(numStates + 1);
        s_a_s_a_counts[i].resize(numStates + 1);

        for (int j = 0; j < numStates + 1; j++) {
            s_a_s_counts[i][j] = MatrixXd::Zero(numActions, numStates + 1);
            s_a_s_r_sums[i][j] = MatrixXd::Zero(numActions, numStates + 1);
            s_a_s_a_counts[i][j].resize(numActions);
            for (int a = 0; a < numActions; a++){
                s_a_s_a_counts[i][j][a] = MatrixXd::Zero(numStates + 1, numActions);
            }
        }
    }
    int s, a, sPrime, aPrime;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        sPrime = (t == traj.len - 1) ? numStates : traj.states[t + 1];
        aPrime = (t == traj.len - 1) ? 0 : traj.actions[t + 1];

        int time_index = 0;
        if (L != 1) {
            time_index = t;
        }
        s_counts[time_index](s) += 1;
        s_a_counts[time_index](s, a) += 1;
        s_a_s_counts[time_index][s](a, sPrime) += 1;
        s_a_s_r_sums[time_index][s](a, sPrime) += traj.rewards[t];
        s_a_s_a_counts[time_index][s][a](sPrime, aPrime) += 1;
      }
    } 
}

void Model::loadEvalPolicy(const Policy & pi, const int & L) {
	//cout << "Starting..." << endl;
	// Load actionProbabilities - so we only compute them once
	actionProbabilities.resize(numStates);
	for (int s = 0; s < numStates; s++){
		actionProbabilities[s] = pi.getActionProbabilities(s);
		// cout << s << " " << actionProbabilities[s] << endl;
	}
	// Q[t](s,a) = Q(s,a) given that S_t=s, A_t=a, and at t=L the state is absorbing.
	vector<VectorXd> temp_V;
    Q.resize(L);
	V.resize(L);
    temp_V.resize(L);
    for (int t = L-1; t >= 0; t--) {
		Q[t] = MatrixXd::Zero(numStates+1, numActions);
        V[t] = VectorXd::Zero(numStates + 1);
        temp_V[t] = VectorXd::Zero(numStates + 1);
        while (true) {
            //cout << "Time step " << t << endl;
            for (int s = 0; s < numStates; s++) {
                for (int a = 0; a < numActions; a++) {
                    double res = 0;
                    for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
                        //Q[t](s,a) += P[s][a][sPrime] * R[s][a][sPrime];
                        res += P[s][a][sPrime] * R[s][a][sPrime];
                        if (L != 1) {
                            if ((sPrime != numStates) && (t != L-1))
                                Q[t](s, a) += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t+1].row(sPrime));
                        } else {
                            if ((sPrime != numStates))
                                // t = 0 always in this case
                                // Q[t](s, a) += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t].row(sPrime));
                                res += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t].row(sPrime));
                        }
                    }
                    Q[t](s,a) = res;
                }
            }
            temp_V[t] = V[t];
            int num = 0;
            for (int s = 0; s < numStates; s++) {
                V[t][s] = actionProbabilities[s].dot(Q[t].row(s));
                if (abs(temp_V[t][s] - V[t][s]) <= eps) num++;
            }
            if (num == numStates) {
                std::cout << "true MDP converged" << std::endl;
                break;
            }
        }
	}
    /*
    for (int t = L-1; t >= 0; t--) {
		Q[t] = MatrixXd::Zero(numStates+1, numActions);
		//cout << "Time step " << t << endl;
		for (int s = 0; s < numStates; s++) {
			for (int a = 0; a < numActions; a++) {
				for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
					Q[t](s,a) += P[s][a][sPrime] * R[s][a][sPrime];
		            if (L != 1) {
                        if ((sPrime != numStates) && (t != L-1))
						    Q[t](s, a) += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t+1].row(sPrime));
                    } else {
                        if ((sPrime != numStates))
                            // t = 0 always in this case
						    Q[t](s, a) += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t].row(sPrime));
                    }
				}
			}
		}
	}
    */
    

    /*
	// Load V[t](s) = V(s) given that S_t = s and at t=L the state is absorbing.
	for (int t = 0; t < L; t++) {
		V[t] = VectorXd::Zero(numStates + 1);
		for (int s = 0; s < numStates; s++)
			V[t][s] = actionProbabilities[s].dot(Q[t].row(s));
	}
    */
	// Load Rsa, Rsa[t](s,a) = Prediction of R_t given that S_0=s and A_0=a
	/*
	Rsa.resize(L);
	for (int i = 0; i < L; i++)
		Rsa[i] = MatrixXd::Zero(numStates+1, numActions);
	for (int initialState = 0; initialState < numStates; initialState++) {
		for (int initialAction = 0; initialAction < numActions; initialAction++) {
			VectorXd stateDistribution = VectorXd::Zero(numStates+1);
			stateDistribution[initialState] = 1;
			for (int t = 0; t < L; t++) {
				VectorXd newStateDistribution = VectorXd::Zero(numStates+1);
				newStateDistribution[numStates] = stateDistribution[numStates];
				for (int s = 0; s < numStates; s++) {
					for (int a = 0; a < numActions; a++) {
						if ((t == 0) && (a != initialAction))
							continue; // Fix the first action
						for (int sPrime = 0; sPrime < numStates+1; sPrime++) {
							double transitionProbability = stateDistribution[s]*(t == 0 ? 1 : actionProbabilities[s][a])*P[s][a][sPrime];
							Rsa[t](initialState, initialAction) += transitionProbability*R[s][a][sPrime];
							newStateDistribution[sPrime] += transitionProbability;
						}
					}
				}
				stateDistribution = newStateDistribution;
			}
		}
	}

	// Load Rs(s,t) = Prediction of R_t given that S_0 = s
	Rs = MatrixXd::Zero(numStates+1, L);
	for (int initialState = 0; initialState < numStates; initialState++) {
		for (int t = 0; t < L; t++)
			Rs(initialState, t) = actionProbabilities[initialState].dot(Rsa[t].row(initialState));
	}
	*/

	evalPolicyValue = 0;
	for (int s = 0; s < numStates; s++)
		evalPolicyValue += d0[s]*V[0][s];

	//cout << "done..." << endl;
}

void Model::load_Q_DP(const Policy & pi, const int & L, vector<Trajectory> &all_data, string dp_type) {
	//cout << "Starting..." << endl;
	// Load actionProbabilities - so we only compute them once
	actionProbabilities.resize(numStates);

    compute_CEE_MDP(L, all_data);
	for (int s = 0; s < numStates; s++){
        if (dp_type.compare("cee-exp-sarsa-mdp") == 0){
		    actionProbabilities[s] = pi.getActionProbabilities(s);
        } else if (dp_type.compare("cee-sarsa-mdp") == 0){
	        VectorXd temp = VectorXd::Zero(numActions);
	        for (int a = 0; a < numActions; a++){
                if (s_counts[0](s) != 0)
		            temp[a] = (s_a_counts[0](s, a) / s_counts[0](s));
            }
            actionProbabilities[s] = temp; 
        }
        else if (dp_type.compare("cee-psec-sarsa-mdp") == 0){
	        VectorXd temp = pi.getActionProbabilities(s);
	        for (int a = 0; a < numActions; a++){
                // use true probabilities, but if this action wasnt taken in this
                // state according to batch, discard it (note pi wont necessarily be supported anymore)
                if (s_a_counts[0](s, a) == 0){
                    //temp[a] = 0.;
                }
            }
            actionProbabilities[s] = temp; 
        }
		// cout << s << " " << actionProbabilities[s] << endl;
	}
	// Q[t](s,a) = Q(s,a) given that S_t=s, A_t=a, and at t=L the state is absorbing.
	
	Q_cee_mdp.resize(L);
    vector<MatrixXd> temp_Q;
    temp_Q.resize(L);
    vector<VectorXd> temp_V;
    temp_V.resize(L);
	V_cee_mdp.resize(L);
	
    for (int t = L-1; t >= 0; t--) {
		Q_cee_mdp[t] = MatrixXd::Zero(numStates+1, numActions);
        temp_Q[t] = MatrixXd::Zero(numStates+1, numActions);
        temp_V[t] = VectorXd::Zero(numStates + 1);
		V_cee_mdp[t] = VectorXd::Zero(numStates + 1);

        while (true) {
            temp_Q[t] = Q_cee_mdp[t];
            //cout << "Time step " << t << endl;
            for (int s = 0; s < numStates; s++) {
                for (int a = 0; a < numActions; a++) {
                    double res = 0;
                    if (s_a_counts[t](s, a) == 0) {
                        continue;
                    }
                    double pi_est = s_a_counts[t](s, a) / s_counts[t](s);
                    for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
                        //if (s_a_s_counts[t][s](a, sPrime) == 0) {
                        //    continue;
                        //}
                        
                        double trans_est = double(s_a_s_counts[t][s](a, sPrime)) / s_a_counts[t](s, a);
                        //double reward_est = double(s_a_s_r_sums[t][s](a, sPrime)) / s_a_s_counts[t][s](a, sPrime); 
                        double reward_est = R[s][a][sPrime];
                        res += trans_est * reward_est;
                        
                        //Q_cee_mdp[t](s,a) += trans_est * reward_est;
                        if (L != 1) {
                            if ((sPrime != numStates) && (t != L-1))
                                Q_cee_mdp[t](s, a) += trans_est * actionProbabilities[sPrime].dot(Q_cee_mdp[t+1].row(sPrime));
                        } else {
                            if ((sPrime != numStates)){
                                // t = 0 always in this case
                                //std::cout << "act probs of next state" <<std::endl;
                                //std::cout << actionProbabilities[sPrime] << std::endl;
                                //res += trans_est * actionProbabilities[sPrime].dot(Q_cee_mdp[t].row(sPrime));
                                for (int aprime = 0; aprime < numActions; aprime++){
                                    
                                    // sarsa fixed point below
                                    //res += trans_est * (s_a_s_a_counts[t][s][a](sPrime, aprime)/s_a_s_counts[t][s](a, sPrime)) * Q_cee_mdp[t](sPrime, aprime);
                                    //PSEC-SARSA-ESTIMTATE fixed point
                                    if (s_a_s_a_counts[t][s][a](sPrime, aprime) != 0) {
                                        res += trans_est * actionProbabilities[sPrime][aprime] * Q_cee_mdp[t](sPrime, aprime);
                                    }
                                }
                            }
                        }
                    }
                    Q_cee_mdp[t](s, a) = res;
                }
            }
            int num = 0;
            temp_V[t] = V_cee_mdp[t];
            
            // Load V[t](s) = V(s) given that S_t = s and at t=L the state is absorbing.
            for (int s = 0; s < numStates; s++) {
                for (int a = 0; a < numActions; a++){
                    if(abs(temp_Q[t](s,a) - Q_cee_mdp[t](s,a)) <= eps) num++;
                }
                //V_cee_mdp[t][s] = actionProbabilities[s].dot(Q_cee_mdp[t].row(s));
                //if (abs(temp_V[t](s) - V_cee_mdp[t](s)) <= eps) num++;
            }
            if (num == numStates * numActions) {
                std::cout << "CEE MDP converged" << std::endl;
                break;
            }
        }
	}
}

vector<Trajectory> Model::generateTrajectories(const Policy & pi, int N, mt19937_64 & generator) const {
	vector<VectorXd> _pi(numStates);
	for (int s = 0; s < numStates; s++)
		_pi[s] = pi.getActionProbabilities(s);

	uniform_real_distribution<double> distribution(0,1);
	vector<Trajectory> trajs(N);
	for (int i = 0; i < N; i++) {
		trajs[i].states.resize(0);
		trajs[i].actions.resize(0);
		trajs[i].rewards.resize(0);
		trajs[i].actionProbabilities.resize(0);
		trajs[i].R = 0;

		int state, action, newState;
		double reward;

		// Get initial state
		state = wrand(generator, d0);
		//cout << "Start: " << state << endl;
		for (int t = 0; true; t++) {
			// Get action
			action = wrand(generator, _pi[state]);

			// Get next state
			newState = wrand(generator, P[state][action]);

			// Get reward
			reward = R[state][action][newState];

			// Add to return
			trajs[i].R += reward;

			// Store in traj
			trajs[i].states.push_back(state);
			trajs[i].actions.push_back(action);
			trajs[i].rewards.push_back(reward);
			trajs[i].actionProbabilities.push_back(_pi[state][action]);

			// Check if episode over
			if (L == 1) {
			    if ((newState == numStates)) {
				    trajs[i].len = t+1;
				    break;
			    }
            } else {
			    if ((newState == numStates) || (t == L-1)) {
				    trajs[i].len = t+1;
				    break;
			    }
            }
			// Set state <-- next-state.
			state = newState;
		}
//		cout << "End: " << trajs[i].len << endl;
	}
	return trajs;
}

double Model::evalMonteCarlo(const Policy & pi, int N, mt19937_64 & generator) const {

	vector<Trajectory> trajs = generateTrajectories(pi, N, generator);
	double R = 0.0;
	for (int i=0; i < trajs.size(); i++) {
          for (int t=0; t<trajs[i].rewards.size(); t++) {
		R += trajs[i].rewards[t];
		//cout << trajs[i].rewards[t] << endl;
	  }
	}
	return R / trajs.size();
}	
