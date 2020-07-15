#include <iostream>
#include "batch_methods.h"

BatchTD::BatchTD(int num_states, int max_traj_len, float lambda, float alpha, bool rho_new_est) {
    //std::cout << "BatchTD constructor..." << std::endl;
	rho_new_est_ = rho_new_est;
    lambda_ = lambda;
	alpha_ = alpha;
	num_states_ = num_states;
    max_traj_length_ = max_traj_len;
	v_.resize(max_traj_len, num_states);
    v_old_.resize(max_traj_len, num_states);
    v_accum_.resize(max_traj_len, num_states);
	e_.resize(max_traj_len, num_states);
	v_grad_.resize(max_traj_len, num_states);
	for (int t=0; t < max_traj_len; t++) {
		for (int s=0; s < num_states; s++) {
			v_(t, s) = 0;
            v_old_(t, s) = 0;
			e_(t, s) = 0;
			v_grad_(t, s) = 0;
            v_accum_(t, s) = 0;
		}
	}
}

void BatchTD::flush_accumulation() {
    //std::cout << "BatchTD flush_accumulation..." << std::endl;
    v_accum_.setZero(v_accum_.rows(), v_accum_.cols());
}

bool BatchTD::accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) {
	
    //std::cout << "BatchTD accumulate..." << std::endl;
    double delta;
    double new_estimate;
    int c_time = t;
    int n_time = t + 1;

    if (max_traj_length_ == 1) {
        c_time = 0;
        n_time = 0;
    }

    //std::cout << "reward " << std::endl;
    //std::cout << R << std::endl;
    if (sPrime == -1) new_estimate = R;
    else new_estimate = R + v_(n_time, sPrime);

    // IW mutliplication only on return estimate (based off of 
    // http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf
    // slide: 34)
    double td_error = new_estimate - v_(c_time, s);

    if (rho_new_est_) {
        delta = IW * new_estimate - v_(c_time, s);
    } else {
        delta = IW * td_error;
    }

    if (sPrime != -1 && abs(td_error) >= max_td_error) {
        max_td_error = abs(td_error);
    }
    
    v_grad_(c_time, s) = 1;
	e_ = lambda_ * e_ + v_grad_;

    // not updating actual function, accumulating TD error
    // IW = 1 if regular TD(0), IW != 1 if using RIS or using another
    // behavior policy	
    v_accum_ += alpha_ * delta * e_;
    //std::cout << "old estimate: " + std::to_string(v_(t,s)) << std::endl;
    //std::cout << "new estimate: " + std::to_string(IW * new_estimate) << std::endl;
    //std::cout << "increment: " + std::to_string(delta) << std::endl;
	v_grad_(c_time, s) = 0;

    // returning true if diverged
    return isnan(v_(c_time, s)) or isinf(v_(c_time, s));
}

void BatchTD::update() {
    //std::cout << "BatchTD update..." << std::endl;
    v_old_ = v_;
	v_ += v_accum_;	
}

void BatchTD::compute_RIS_estimates(vector<Trajectory> all_data) {}

bool BatchTD::has_converged(bool td_error_check, double eps) {

  if (td_error_check) {
    
    if(max_td_error < eps) {
        std::cout << "breaking... max td error is " << std::to_string(max_td_error) << std::endl;
        max_td_error = std::numeric_limits<double>::min(); 
        return true;
    }
    max_td_error = std::numeric_limits<double>::min(); 
    return false;
  } else { 
      int total_c = 0;
      int converged = 0; 
      for (int t=0; t < max_traj_length_; t++) {
        for (int s=0; s < num_states_; s++) {

          // only checking convergence for states that have been updated (!= 0)
          if (v_(t, s) != 0) {
            total_c += 1;
            if (abs(v_(t, s) - v_old_(t, s)) <= eps) {
                converged += 1;
            }
          }
        }
      }
    return (converged == total_c);
  } 
}

void BatchTD::compute_traj_RIS_IWs(vector<Trajectory> & trajs, const Policy &eval_pi) {}

void BatchTD::update_learning_rate(double new_lr) {
    alpha_ = new_lr;
}

BatchRISTD::BatchRISTD(Policy pi, int num_states, int num_actions, int max_traj_len, float lambda, float alpha, bool rho_new_est) {
    //std::cout << "BatchRISTD constructor..." << std::endl;
    
    // storing evaluation policy (typically)
    policy_ = pi;
	rho_new_est_ = rho_new_est;
    // for RIS estimates
	//state_action_counts_.resize(num_states, num_actions);
	//state_counts_.resize(num_states);
	
	// time version, uncomment above for non-time index
    state_action_counts_.resize(max_traj_len);
    state_counts_.resize(max_traj_len);
    for (int i = 0; i < max_traj_len; i++){
        state_action_counts_[i] = MatrixXd::Zero(num_states + 1, num_actions);
        state_counts_[i] = VectorXd::Zero(num_states + 1);
    }

    // TD specifics
    lambda_ = lambda;
	alpha_ = alpha;
	num_states_ = num_states;
    max_traj_length_ = max_traj_len;
	v_.resize(max_traj_len, num_states);
    // maintaing old v for convergence check
    v_old_.resize(max_traj_len, num_states);
    // accumulation for batch
    v_accum_.resize(max_traj_len, num_states);
	e_.resize(max_traj_len, num_states);
	v_grad_.resize(max_traj_len, num_states);
	for (int t=0; t < max_traj_len; t++) {
		for (int s=0; s < num_states; s++) {
			v_(t, s) = 0;
            v_old_(t, s) = 0;
			e_(t, s) = 0;
			v_grad_(t, s) = 0;
            v_accum_(t, s) = 0;
            
            // non-timed based init
            //state_counts_(s) = 0;
            //for (int a = 0; a < num_actions; a++) {
            //    state_action_counts_(s, a) = 0;
            //}
		}
	}
	// std::cout << "RisTD constructor" << std::endl;
}

void BatchRISTD::flush_accumulation() {
    //std::cout << "BatchRISTD flush_accumulation..." << std::endl;
    v_accum_.setZero(v_accum_.rows(), v_accum_.cols());
}

bool BatchRISTD::accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) {

    // IMP: IW is ignored here. Just passed in for simplicity from OOP point of view
    // But IW is re-calculated for RIS specifically (rho)	
    double delta;
    double new_estimate;
    //double rho = IW;
    
    int c_time = t;
    int n_time = t + 1;

    if (max_traj_length_ == 1) {
        c_time = 0;
        n_time = 0;
    }   
    double rho = policy_.getActionProbability(s, a) / state_action_counts_[c_time](s, a) * state_counts_[c_time](s);

    //std::cout << "rho " << std::to_string(rho) << std::endl;
    if (sPrime == -1) new_estimate = R;
    else new_estimate = R + v_(n_time, sPrime);

    // IW mutliplication only on return estimate (based off of 
    // http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf
    // slide: 34)
    
    if (rho_new_est_) {
        delta = rho * new_estimate - v_(c_time,s);
    } else {
        delta = rho * (new_estimate - v_(c_time,s));
    }
    
    v_grad_(c_time, s) = 1;
	e_ = lambda_ * e_ + v_grad_;

    // not updating actual function, accumulating TD error
    // IW = 1 if regular TD(0), IW != 1 if using RIS or using another
    // behavior policy	
    v_accum_ += alpha_ * delta * e_;
    //std::cout <<  "rho " << std::to_string(rho) << std::endl;
	//std::cout << "old estimate: " + std::to_string(v_(t,s)) << std::endl;
    //std::cout << "new estimate: " + std::to_string(rho * new_estimate) << std::endl;
    //std::cout << "increment: " + std::to_string(delta) << std::endl;

    v_grad_(c_time, s) = 0;

    // return true if diverged
    return isnan(v_(c_time, s)) or isinf(v_(c_time, s));
}

void BatchRISTD::update() {
    //std::cout << "BatchRISTD update..." << std::endl;
	v_old_ = v_;
    v_ += v_accum_;	
}

void BatchRISTD::compute_RIS_estimates(vector<Trajectory> all_data) {

    //std::cout << "BatchRISTD compute_RIS_estimates..." << std::endl;
    int s, a;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        state_action_counts_[0](s, a) += 1;
        state_counts_[0](s) += 1;

        //state_action_counts_[t](s, a) += 1;
        //state_counts_[t](s) += 1;
      }
    } 
}

bool BatchRISTD::has_converged(bool td_error_check, double eps) {
 
  int total_c = 0;
  int converged = 0; 
  for (int t=0; t < max_traj_length_; t++) {
    for (int s=0; s < num_states_; s++) {

      // only checking convergence for states that have been updated (!= 0)
      if (v_(t, s) != 0) {
        total_c += 1;
        if (abs(v_(t, s) - v_old_(t, s)) <= eps) {
            converged += 1;
        }
      }
    }
  }  
  return (converged == total_c);
}

void BatchRISTD::compute_traj_RIS_IWs(vector<Trajectory> & trajs, const Policy &eval_pi) {
  
  vector<double> iws;
  int s, a;
  for (int i = 0; i < static_cast<int>(trajs.size()); i++) {
    trajs[i].IWs.resize(trajs[i].len);
    trajs[i].cumIWs.resize(trajs[i].len);
    trajs[i].evalActionProbabilities.resize(trajs[i].len);
    for (int t = 0; t < trajs[i].len; t++) {
      //trajs[i].evalActionProbabilities[t] = eval_pi.getActionProbability(
      //  trajs[i].states[t], trajs[i].actions[t]);
      s = trajs[i].states[t];
      a = trajs[i].actions[t];
      trajs[i].IWs[t] = eval_pi.getActionProbability(trajs[i].states[t],
                trajs[i].actions[t]) / state_action_counts_[t](s, a) * state_counts_[t](s);
      //trajs[i].evalActionProbabilities[t] /
      //trajs[i].actionProbabilities[t];
      //std::cout << "IW : " << std::to_string(trajs[i].IWs[t]) << std::endl;
      trajs[i].cumIWs[t] = trajs[i].IWs[t];
      if (t != 0)
        trajs[i].cumIWs[t] *= trajs[i].cumIWs[t-1];
      iws.push_back(trajs[i].cumIWs[t]);
    }
  }
  sort(iws.begin(), iws.end());
    for (int i = 0; i < iws.size(); i++) {
        std::cout << iws.at(i) << ' ';
    }
}

void BatchRISTD::update_learning_rate(double new_lr) {
    alpha_ = new_lr;
}

// SARSA
BatchSARSA::BatchSARSA(Policy pi, int num_states, int num_actions, int max_traj_len, float lambda, float alpha, bool expected_sarsa, bool ris, bool rho_new_est) {
    //std::cout << "BatchRISTD constructor..." << std::endl;
    
    // storing evaluation policy (typically)
    policy_ = pi;
	exp_sarsa_ = expected_sarsa;
    ris_ = ris;
    rho_new_est_ = rho_new_est;
    // for RIS estimates
    state_action_counts_.resize(max_traj_len);
    state_counts_.resize(max_traj_len);
    s_a_s_counts_.resize(max_traj_len);
    s_a_s_a_counts_.resize(max_traj_len);
    for (int i = 0; i < max_traj_len; i++){
        state_action_counts_[i] = MatrixXd::Zero(num_states + 1, num_actions);
        state_counts_[i] = VectorXd::Zero(num_states + 1);
        
        s_a_s_counts_[i].resize(num_states + 1);
        s_a_s_a_counts_[i].resize(num_states + 1);
        for (int j = 0; j < num_states + 1; j++) {
            s_a_s_counts_[i][j] = MatrixXd::Zero(num_actions, num_states + 1);
            s_a_s_a_counts_[i][j].resize(num_actions);
            for (int a = 0; a < num_actions; a++){
                s_a_s_a_counts_[i][j][a] = MatrixXd::Zero(num_states + 1, num_actions);
            }
        }
    }

    // TD specifics
    lambda_ = lambda;
	alpha_ = alpha;
	num_states_ = num_states;
    num_actions_ = num_actions;
    max_traj_length_ = max_traj_len;

    Q_.resize(max_traj_len);
    Q_old_.resize(max_traj_len);
    Q_accum_.resize(max_traj_len);
    actionProbabilities.resize(num_states);
    for (int s = 0; s < num_states; s++) {
        actionProbabilities[s] = pi.getActionProbabilities(s);
    }
	v_.resize(max_traj_len, num_states);
    // maintaing old v for convergence check
    v_old_.resize(max_traj_len, num_states);
    // accumulation for batch
    v_accum_.resize(max_traj_len, num_states);
	e_.resize(max_traj_len, num_states);
	v_grad_.resize(max_traj_len, num_states);
	for (int t=0; t < max_traj_len; t++) {
		for (int s=0; s < num_states; s++) {
			v_(t, s) = 0;
            v_old_(t, s) = 0;
			e_(t, s) = 0;
			v_grad_(t, s) = 0;
            v_accum_(t, s) = 0;
		}
        Q_[t] = MatrixXd::Zero(num_states, num_actions);
        Q_old_[t] = MatrixXd::Zero(num_states, num_actions);
        Q_accum_[t] = MatrixXd::Zero(num_states, num_actions);
	}
}

void BatchSARSA::flush_accumulation() {
    for (int t = 0; t < max_traj_length_; t++){
        Q_accum_[t].setZero(Q_accum_[t].rows(), Q_accum_[t].cols());
    }
}

bool BatchSARSA::accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) {

    // IMP: IW is ignored here. Just passed in for simplicity from OOP point of view
    // But IW is re-calculated for RIS specifically (rho)	
    double delta;
    double new_estimate;
    //double rho = IW;
    
    int c_time = t;
    int n_time = t + 1;

    if (max_traj_length_ == 1) {
        c_time = 0;
        n_time = 0;
    }   

    //std::cout << "rho " << std::to_string(rho) << std::endl;
    if (sPrime == -1) { 
        new_estimate = R;
        delta = new_estimate - Q_[c_time](s, a);
    }
    else {
        if (exp_sarsa_){
            new_estimate = R + actionProbabilities[sPrime].dot(Q_[n_time].row(sPrime));
        } else {
            new_estimate = R + Q_[n_time](sPrime, aPrime);
        }
        if (ris_){
            //double rho = policy_.getActionProbability(sPrime, aPrime) / state_action_counts_[n_time](sPrime, aPrime) * state_counts_[n_time](sPrime);
            double rho = policy_.getActionProbability(sPrime, aPrime) / s_a_s_a_counts_[n_time][s][a](sPrime, aPrime) * s_a_s_counts_[n_time][s](a, sPrime);
            if (rho_new_est_){
                delta = rho * new_estimate - Q_[c_time](s, a);
            } else {
                delta = rho * (new_estimate - Q_[c_time](s, a));
            }
        } else {
            delta = new_estimate - Q_[c_time](s, a);
        }
        //std::cout << "exp sarsa " << exp_sarsa_ << " ris " << ris_ << " rho new est " << rho_new_est_ << std::endl;
        //std::cout << "finished boostrapping next state, updating now 1 " <<std::endl;
    }
    
    Q_accum_[c_time](s, a) += alpha_ * delta;
    // return true if diverged
    return isnan(Q_[c_time](s)) or isinf(Q_[c_time](s));
}

void BatchSARSA::update() {
    //std::cout << "BatchSARSA update..." << std::endl;
	v_old_ = v_;
	for (int t = 0; t < max_traj_length_; t++){
        Q_old_[t] = Q_[t];
        Q_[t] += Q_accum_[t];
        for (int s = 0; s < num_states_; s++){
            v_(t, s) = actionProbabilities[s].dot(Q_[t].row(s));
        }
    }
    //std::cout << "BatchSARSA finished..." << std::endl;
}

bool BatchSARSA::has_converged(bool td_error_check, double eps) {
  int total_c = 0;
  int converged = 0; 
  for (int t=0; t < max_traj_length_; t++) {
    for (int s=0; s < num_states_; s++) {
        for (int a=0; a < num_actions_; a++){

          // only checking convergence for states that have been updated (!= 0)
          if(Q_[t](s, a) != 0) { /*if (v_(t, s) != 0) {*/
            total_c += 1;
            if (abs(Q_[t](s, a) - Q_old_[t](s, a)) <= eps) {/*if (abs(v_(t, s) - v_old_(t, s)) <= eps) {*/
                converged += 1;
            }
          }
        }
    }
  }  
  return (converged == total_c);
}

void BatchSARSA::compute_RIS_estimates(vector<Trajectory> all_data) {

    //std::cout << "BatchRISTD compute_RIS_estimates..." << std::endl;
    int s, a, sPrime, aPrime;
    for (auto & traj : all_data) {
      for (int t=0; t < traj.len; t++) {
        s = traj.states[t];
        a = traj.actions[t];
        sPrime = (t == traj.len - 1) ? num_states_ : traj.states[t + 1];
        aPrime = (t == traj.len - 1) ? 0 : traj.actions[t + 1];
        
        //s_counts_[0](s) += 1;
        //s_a_counts_[0](s, a) += 1;
        s_a_s_counts_[0][s](a, sPrime) += 1;
        s_a_s_a_counts_[0][s][a](sPrime, aPrime) += 1;
        //state_action_counts_[t](s, a) += 1;
        //state_counts_[t](s) += 1;
      }
    } 
}

void BatchSARSA::compute_traj_RIS_IWs(vector<Trajectory> & trajs, const Policy &eval_pi) {}

void BatchSARSA::update_learning_rate(double new_lr) {
    alpha_ = new_lr;
}
