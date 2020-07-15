#include "experiments.h"
#include "utils.h"
#include "results.pb.h"
#include "batch_methods.h"
#include <math.h>

void BatchEvaluate(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs,
                       int seed, string outfile, int method, string method_name,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha,
                       bool rho_new_est) {

  std::cout << "in BatchEvaluate...\n" << std::endl;
  bool on_policy = (target_policy_number == behavior_policy_number);
  
  Policy pi = env->getPolicy(target_policy_number);
  Policy behavior_pi = env->getPolicy(behavior_policy_number);
  vector<Trajectory> data;
  vector<Trajectory> all_data;
  mt19937_64 generator(seed);

  evaluation::MethodResult result_proto;
 
  BatchVFMethod* batch_vf_method;
  float lambda = 0.0;

  if (alpha == -1) {
    std::cout << "original learning rate -1, setting default" << std::endl;
    alpha = 1.0 / double(num_trajs * env->getNumStates());
  }
  double eps = 1e-10; //1e-8; //1e-15;

  if (method == 0) {
    batch_vf_method = new BatchRISTD(pi, env->getNumStates(), env->getNumActions(),
                                    env->getMaxTrajLen(), lambda, alpha, rho_new_est);
  } else if (method == 1) {
    batch_vf_method = new BatchTD(env->getNumStates(), env->getMaxTrajLen(),
                                    lambda, alpha, rho_new_est);
  } 
  result_proto.set_method_name(method_name);

  std::cout << "collecting batch data..." << std::endl;
  // collecting num_trajs trajectories
  // contains the probability of taking an action in a given state
  // according to the behavior_pi
  env->generateTrajectories(data, behavior_pi, num_trajs, generator);

  // given that data has the true behavior policy weights
  // and that we have the true evaluation policy, below we calculate the
  // importance weights

  // IMP: calculates the IWs based on true probablitiy distribution
  // regardless if on- or off-policy, so calculates IW for a particular
  // trajectory at a particular time t
  // 
  // RIS estimate is done separately
  
  // always computed, but may be overriden
  LoadTrajIWs(data, pi);
  for (auto & traj : data) {
    all_data.push_back(traj);
  }

  // used by DP for true value function estimations
  std::cout << "getting env model..." << std::endl;
  Model m = env->getTrueModel();

  std::cout << "got env model..." << std::endl;
  // TODO: change the strucutre of this calling
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "loading true mdp" << std::endl;
    m.loadEvalPolicy(pi, env->getMaxTrajLen());
  } else if (dp_type.compare("cee-mrp") == 0) {
    std::cout << "loading cee mrp" << std::endl;
    m.load_DP_CEE_MRP(env->getMaxTrajLen(), all_data);
  } else if (dp_type.compare("cee-mdp") == 0) {
    std::cout << "loading cee mdp" << std::endl;
    m.load_DP_CEE_MDP(pi, env->getMaxTrajLen(), all_data);
  } 

  if (method == 0) {
    std::cout << "computing RIS estimates..." << std::endl;
    batch_vf_method->compute_RIS_estimates(all_data);
    //batch_vf_method->compute_traj_RIS_IWs(all_data, pi);
  }

  int batch_process_step = 1;
  bool diverged = false;
  int s, a, sPrime, aPrime;
  double R;
  double msve;
  double old_msve = -1;
  std::cout << "starting value function learning..." << std::endl;
  while (true /*batch_process_step < batch_process_steps*/) {
    //std::cout << "batch processing step " + std::to_string(batch_process_step) << std::endl;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        R = traj.rewards[t];
        if (t < traj.len - 1) {
            sPrime = traj.states[t + 1];
            aPrime = traj.actions[t + 1];
        }
        else {
            sPrime = -1;
            aPrime = -1;
        }

        // accumulate increments
        // depending on the type of algorithm traj.IWs[t] may not be used. For example,
        // not used for RIS
        //batch_vf_method->accumulate(t, s, a, R, sPrime, traj.cumIWs[t]);//traj.IWs[t]);
        diverged = batch_vf_method->accumulate(t, s, a, R, sPrime, aPrime, traj.IWs[t]);
      }
    }

    // update value function with accumulations
    batch_vf_method->update();

    // clear out earlier increment accumulation
    batch_vf_method->flush_accumulation();

    //batch_vf_method->update_learning_rate(1.0 / float(num_trajs * env->getNumStates() * log10(batch_process_step + 1)));
    /*
    if (compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(),
                            env->getNumStates(), dp_type) == 0.0) {
        std::cout << "breaking at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    */
    if (diverged) {
        std::cout << "diverged at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    if (batch_vf_method->has_converged(false, eps)) {
        std::cout << "breaking at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    if (batch_process_step % 100 == 0) {
        std::cout << "batch processing step " + std::to_string(batch_process_step)  << std::endl;
        std::cout << batch_vf_method->v_  << std::endl;
    }
    if (batch_process_step % 10 == 0) {
        result_proto.add_batch_process_num_steps(batch_process_step);
        result_proto.add_batch_process_mses(compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(), env->getNumStates(), dp_type));
    }
    batch_process_step += 1;
  }

  // checking mean-squared-value-error after convergence
  msve = compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(),
                            env->getNumStates(), dp_type);
  result_proto.add_value_error(msve);
  result_proto.add_num_steps_observed(num_trajs);

  float visited = (float) get_visited_s_a_count(all_data, env->getNumStates(), env->getNumActions());
  float unvisited_s_a_ratio = 1.0 - (visited / (env->getNumStates() * env->getNumActions()));
  result_proto.add_num_unvisited_s_a(unvisited_s_a_ratio);
  result_proto.add_deterministic_prob(deterministic_prob);
  delete batch_vf_method;

  fstream output(outfile, ios::out | ios::trunc | ios::binary);
  if (!result_proto.SerializeToOstream(&output)) {
    cerr << "Failed to write results." << endl;
  }
  std::cout << "finished BatchEvaluate with: num trajectories: " + std::to_string(num_trajs) + " and MSVE: " << std::scientific << msve <<  " and unvisited (s,a) ratio: " << unvisited_s_a_ratio << std::endl;
}

void BatchEvaluateSA(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs,
                       int seed, string outfile, int method, string method_name, bool ris,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha,
                       bool rho_new_est, bool exp_sarsa) {

  std::cout << "in BatchEvaluate...\n" << std::endl;
  bool on_policy = (target_policy_number == behavior_policy_number);
  
  Policy pi = env->getPolicy(target_policy_number);
  Policy behavior_pi = env->getPolicy(behavior_policy_number);
  vector<Trajectory> data;
  vector<Trajectory> all_data;
  mt19937_64 generator(seed);

  evaluation::MethodResult result_proto;
 
  BatchVFMethod* batch_vf_method;
  float lambda = 0.0;

  if (alpha == -1) {
    std::cout << "original learning rate -1, setting default" << std::endl;
    alpha = 1.0 / double(num_trajs * env->getNumStates());
  }
  double eps = 1e-5;//1e-10; //1e-8; //1e-15;

  batch_vf_method = new BatchSARSA(pi, env->getNumStates(), env->getNumActions(),
                                    env->getMaxTrajLen(), lambda, alpha, exp_sarsa, ris, rho_new_est);
  result_proto.set_method_name(method_name);

  std::cout << "collecting batch data..." << std::endl;
  // collecting num_trajs trajectories
  // contains the probability of taking an action in a given state
  // according to the behavior_pi
  env->generateTrajectories(data, behavior_pi, num_trajs, generator);

  // given that data has the true behavior policy weights
  // and that we have the true evaluation policy, below we calculate the
  // importance weights

  // IMP: calculates the IWs based on true probablitiy distribution
  // regardless if on- or off-policy, so calculates IW for a particular
  // trajectory at a particular time t
  // 
  // RIS estimate is done separately
  
  // always computed, but may be overriden
  LoadTrajIWs(data, pi);
  for (auto & traj : data) {
    all_data.push_back(traj);
  }

  // used by DP for true value function estimations
  std::cout << "getting env model..." << std::endl;
  Model m = env->getTrueModel();

  std::cout << "got env model..." << std::endl;
  // TODO: change the strucutre of this calling
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "loading true mdp" << std::endl;
    m.loadEvalPolicy(pi, env->getMaxTrajLen());
  } else {
    std::cout << "loading a CEE variant for MDP" << std::endl;
    m.load_Q_DP(pi, env->getMaxTrajLen(), all_data, dp_type);
  }

  if (ris) {
    std::cout << "computing RIS estimates..." << std::endl;
    batch_vf_method->compute_RIS_estimates(all_data);
    //batch_vf_method->compute_traj_RIS_IWs(all_data, pi);
  }

  int batch_process_step = 1;
  bool diverged = false;
  int s, a, sPrime, aPrime;
  double R;
  double msve;
  double old_msve = -1;
  std::cout << "starting value function learning..." << std::endl;
  while (true /*batch_process_step < batch_process_steps*/) {
    //std::cout << "batch processing step " + std::to_string(batch_process_step) << std::endl;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        R = traj.rewards[t];
        if (t < traj.len - 1) {
            sPrime = traj.states[t + 1];
            aPrime = traj.actions[t + 1];
        }
        else {
            sPrime = -1;
            aPrime = -1;
        }

        // accumulate increments
        // depending on the type of algorithm traj.IWs[t] may not be used. For example,
        // not used for RIS
        //batch_vf_method->accumulate(t, s, a, R, sPrime, traj.cumIWs[t]);//traj.IWs[t]);
        diverged = batch_vf_method->accumulate(t, s, a, R, sPrime, aPrime, traj.IWs[t]);
      }
    }

    // update value function with accumulations
    batch_vf_method->update();

    // clear out earlier increment accumulation
    batch_vf_method->flush_accumulation();

    //batch_vf_method->update_learning_rate(1.0 / float(num_trajs * env->getNumStates() * log10(batch_process_step + 1)));
    /*
    if (compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(),
                            env->getNumStates(), dp_type) == 0.0) {
        std::cout << "breaking at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    */
    if (diverged) {
        std::cout << "diverged at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    if (batch_vf_method->has_converged(false, eps)) {
        std::cout << "breaking at " + std::to_string(batch_process_step) << std::endl;
        break;
    }
    if (batch_process_step % 100 == 0) {
        std::cout << "batch processing step " + std::to_string(batch_process_step)  << std::endl;
        //std::cout << batch_vf_method->v_  << std::endl;
    }
    if (batch_process_step % 10 == 0) {
        result_proto.add_batch_process_num_steps(batch_process_step);
        //result_proto.add_batch_process_mses(compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(), env->getNumStates(), dp_type));
        result_proto.add_batch_process_mses(compute_msve_batch_sa(batch_vf_method, m, env->getMaxTrajLen(), env->getNumStates(), env->getNumActions(), dp_type));
    }
    batch_process_step += 1;
  }

  // checking mean-squared-value-error after convergence
  msve = compute_msve_batch_sa(batch_vf_method, m, env->getMaxTrajLen(), env->getNumStates(), env->getNumActions(), dp_type);
  //msve = compute_msve_batch(batch_vf_method, m, env->getMaxTrajLen(),
  //                          env->getNumStates(), dp_type);
  result_proto.add_value_error(msve);
  result_proto.add_num_steps_observed(num_trajs);

  //float visited = (float) get_visited_s_a_count(all_data, env->getNumStates(), env->getNumActions());
  float visited = (float) get_visited_s_a_s_a_count(all_data, env->getNumStates(), env->getNumActions());
  float unvisited_s_a_ratio = 1.0 - (visited / (env->getNumStates() * env->getNumActions() * 1 * env->getNumActions()));
  result_proto.add_num_unvisited_s_a(unvisited_s_a_ratio);
  result_proto.add_deterministic_prob(deterministic_prob);
  delete batch_vf_method;

  fstream output(outfile, ios::out | ios::trunc | ios::binary);
  if (!result_proto.SerializeToOstream(&output)) {
    cerr << "Failed to write results." << endl;
  }
  std::cout << "finished BatchEvaluate with: num trajectories: " + std::to_string(num_trajs) + " and MSVE: " << std::scientific << msve <<  " and unvisited (s,a) ratio: " << unvisited_s_a_ratio << std::endl;
}

double compute_msve_batch_sa(BatchVFMethod* batch_vf_method, Model m,
                            int max_time, int num_states, int num_actions, string dp_type) {
  // std::cout << "MSVE" << std::endl;
  vector<MatrixXd> Q;
  
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "using true mdp for MSE " << std::endl;
    Q = m.Q;
  } else {
    //std::cout << "using  CEE variant for MSE  " << std::endl;
    Q = m.Q_cee_mdp;
  } 

  double error = 0;
  int z_count = 0;
  int c = 0;
  for (int t=0; t < max_time; t++) {
    for (int s=0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++){
            // std::cout << vf_method->v_.rows() << std::endl;
            // std::cout << t << " " << s << std::endl;
            if (batch_vf_method->Q_[t](s,a) == 0) {
                z_count += 1;
            } else {
                error += pow(batch_vf_method->Q_[t](s, a) - Q[t](s, a), 2.f);
                c += 1;
            }
        std::cout << "estimate: " + std::to_string(batch_vf_method->Q_[t](s,a)) << " real: " + std::to_string(Q[t](s,a)) << std::endl;
        }
      
      // std::cout << error << std::endl;
    }
  }
  error = error / c;
  return error;
}

double compute_msve_batch(BatchVFMethod* batch_vf_method, Model m,
                            int max_time, int num_states, string dp_type) {
  // std::cout << "MSVE" << std::endl;
  vector<VectorXd> V;
  
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "using true mdp for MSE " << std::endl;
    V = m.V;
  } else if (dp_type.compare("cee-mrp") == 0) {
    std::cout << "using cee mrp for MSE " << std::endl;
    V = m.V_cee_mrp;
  } else if (dp_type.compare("cee-mdp") == 0) {
    std::cout << "using cee mdp for MSE " << std::endl;
    V = m.V_cee_mdp;
  }  

  double error = 0;
  int z_count = 0;
  int c = 0;
  for (int t=0; t < max_time; t++) {
    for (int s=0; s < num_states; s++) {
      // std::cout << vf_method->v_.rows() << std::endl;
      // std::cout << t << " " << s << std::endl;
      if (batch_vf_method->v_(t,s) == 0) {
        z_count += 1;
      } else {
        error += pow(batch_vf_method->v_(t, s) - V[t](s), 2.f);
        c += 1;
      }
      std::cout << "estimate: " + std::to_string(batch_vf_method->v_(t,s)) << " real: " + std::to_string(V[t](s)) << std::endl;
      
      // std::cout << error << std::endl;
    }
  }

  //std::cout << "univisted states:  " << std::to_string(z_count) << std::endl;
  //std::cout << "total states " << std::to_string(max_time * num_states) << std::endl;
  //error = error / (max_time * num_states);
  error = error / c;
  return error;
}

void LSTDBatchEvaluate(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs,
                       int seed, string outfile, int method, string method_name,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha) {

  std::cout << "in LSTDBatchEvaluate...\n" << std::endl;
  bool on_policy = (target_policy_number == behavior_policy_number);
  
  Policy pi = env->getPolicy(target_policy_number);
  Policy behavior_pi = env->getPolicy(behavior_policy_number);
  vector<Trajectory> data;
  vector<Trajectory> all_data;
  mt19937_64 generator(seed);

  evaluation::MethodResult result_proto;
 
  BatchVFMethod* batch_vf_method;
  float lambda = 0.0;

  if (alpha == -1) {
    std::cout << "original learning rate -1, setting default" << std::endl;
    alpha = 1.0 / double(num_trajs * env->getNumStates());
  }

  result_proto.set_method_name(method_name);

  std::cout << "collecting batch data..." << std::endl;
  // collecting num_trajs trajectories
  // contains the probability of taking an action in a given state
  // according to the behavior_pi
  env->generateTrajectories(data, behavior_pi, num_trajs, generator);

  // given that data has the true behavior policy weights
  // and that we have the true evaluation policy, below we calculate the
  // importance weights

  // IMP: calculates the IWs based on true probablitiy distribution
  // regardless if on- or off-policy, so calculates IW for a particular
  // trajectory at a particular time t
  // 
  // RIS estimate is done separately
  
  // always computed, but may be overriden
  LoadTrajIWs(data, pi);
  for (auto & traj : data) {
    all_data.push_back(traj);
  }

  // used by DP for true value function estimations
  std::cout << "getting env model..." << std::endl;
  Model m = env->getTrueModel();

  std::cout << "got env model..." << std::endl;
  // TODO: change the strucutre of this calling
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "loading true mdp" << std::endl;
    m.loadEvalPolicy(pi, env->getMaxTrajLen());
  } else if (dp_type.compare("cee-mrp") == 0) {
    std::cout << "loading cee mrp" << std::endl;
    m.load_DP_CEE_MRP(env->getMaxTrajLen(), all_data);
  } else if (dp_type.compare("cee-mdp") == 0) {
    std::cout << "loading cee mdp" << std::endl;
    m.load_DP_CEE_MDP(pi, env->getMaxTrajLen(), all_data);
  } 

  MatrixXd s_a_counts = MatrixXd::Zero(env->getNumStates(), env->getNumActions());
  VectorXd s_counts = VectorXd::Zero(env->getNumStates());
  if (method == 0) {
    int s, a;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        s_a_counts(s, a) += 1;
        s_counts(s) += 1;

       }
    }
  } else {
    std::cout << "not PSEC" << std::endl;
  }

  int d = env->getNumStates() + 1;
  MatrixXd A = MatrixXd::Zero(d, d);
  VectorXd b = VectorXd::Zero(d);
   
  int batch_process_step = 1;
  bool diverged = false;
  int s, a, sPrime;
  double R;
  double msve;
  double old_msve = -1;
  double step_count = 0;
  std::cout << "starting value function learning..." << std::endl;
    //std::cout << "batch processing step " + std::to_string(batch_process_step) << std::endl;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        R = traj.rewards[t];
        if (t < traj.len - 1) sPrime = traj.states[t + 1];
        //else sPrime = -1;
        else sPrime = d - 1;


        //if (sPrime == -1) continue;
        step_count++;
        double rho = traj.IWs[t];
        if (method == 0) {
            rho = pi.getActionProbability(s, a) / s_a_counts(s, a) * s_counts(s);
        } 
        std::cout << rho << std::endl;
        // also need feature vectors - one hot
        VectorXd s_feat = VectorXd::Zero(d);
        VectorXd s_prime_feat = VectorXd::Zero(d);
        s_feat(s) = 1.0;
        s_prime_feat(sPrime) = 1.0;
        VectorXd e = rho * s_feat;
        VectorXd diff = s_feat - s_prime_feat;
        A = A + e * diff.transpose();
        b = b + (R * e);
      }
    }
    A = A / step_count;
    b = b / step_count;
    A += (alpha * MatrixXd::Identity(d, d));

  VectorXd w = A.inverse() * b;
  // checking mean-squared-value-error after convergence
  msve = compute_msve_LSTD(w, m, env->getNumStates(), dp_type);
  result_proto.add_value_error(msve);
  result_proto.add_num_steps_observed(num_trajs);

  /*
  float visited = (float) get_visited_s_a_count(all_data, env->getNumStates(), env->getNumActions());
  float unvisited_s_a_ratio = 1.0 - (visited / (env->getNumStates() * env->getNumActions()));
  result_proto.add_num_unvisited_s_a(unvisited_s_a_ratio);
  */
  result_proto.add_deterministic_prob(deterministic_prob);

  fstream output(outfile, ios::out | ios::trunc | ios::binary);
  if (!result_proto.SerializeToOstream(&output)) {
    cerr << "Failed to write results." << endl;
  }
  std::cout << "finished LSTDBatchEvaluate with: num trajectories: " + std::to_string(num_trajs) + " and MSVE: " << std::scientific << msve << std::endl;
  //std::cout << "finished LSTDBatchEvaluate with: num trajectories: " + std::to_string(num_trajs) + " and MSVE: " << std::scientific << msve <<  " and unvisited (s,a) ratio: " << unvisited_s_a_ratio << std::endl;
}

double compute_msve_LSTD(VectorXd w, Model m, int num_states, string dp_type) {
  // std::cout << "MSVE" << std::endl;
  vector<VectorXd> V;
  
  if (dp_type.compare("mdp") == 0) {
    // computes DP using true trans and policy vals
    std::cout << "using true mdp for MSE " << std::endl;
    V = m.V;
  } else if (dp_type.compare("cee-mrp") == 0) {
    std::cout << "using cee mrp for MSE " << std::endl;
    V = m.V_cee_mrp;
  } else if (dp_type.compare("cee-mdp") == 0) {
    std::cout << "using cee mdp for MSE " << std::endl;
    V = m.V_cee_mdp;
  }  

  double error = 0;
  int z_count = 0;
  int c = 0;
    for (int s=0; s < num_states; s++) {
      if (w(s) == 0) {
        z_count += 1;
      } else {
        error += pow(w(s) - V[0](s), 2.f);
        c += 1;
      }
      std::cout << "estimate: " + std::to_string(w(s)) << " real: " + std::to_string(V[0](s)) << std::endl;
    }

  error = error / c;
  return error;
}
