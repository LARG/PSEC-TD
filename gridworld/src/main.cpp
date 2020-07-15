#include <math.h>
#include <iostream>
#include <fstream>

#include <Zaffre/headers/Zaffre.hpp>

#include "utils.h"
#include "experiments.h"

// MDP Stuff
#include "Trajectory.h"
#include "Transition.h"
#include "results.pb.h"

using namespace std;

int main(int nargs, char* args[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int seed = 0;
  int num_trajs = 10;
  int policy_number = 2;
  int behavior_policy_number = 1;
  int print_freq = 10;
  bool ris = false;
  bool cee_mrp = false;
  bool cee_mdp = false;
  bool mdp = false;
  bool cee_exp_sarsa_mdp = false;
  bool cee_sarsa_mdp = false;
  bool cee_psec_sarsa_mdp = false;
  float deterministic_prob = 1.00;
  double alpha = -1;
  bool lstd = false;
  bool rho_new_est = false;
  bool state_action = false;
  bool exp_sarsa = false;
  string input;
  string outfile_name("value.txt");

  for (int i=1; i < nargs; i++) {

    if (strcmp(args[i], "--seed") == 0 && i+1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> seed))
        cout << "Could not parse random seed " << endl;
    }
    
    if (strcmp(args[i], "--outfile") == 0 && i + 1 < nargs)
      outfile_name = args[i+1];
    
    if (strcmp(args[i],"--print_freq") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> print_freq))
        cout << "Could not parse number of iterations to skip" << endl;
    }
    
    if (strcmp(args[i],"--num-trajs") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> num_trajs))
        cout << "Could not parse trajectories per iteration" << endl;
    }
    
    if (strcmp(args[i],"--policy-number") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> policy_number))
        cout << "Could not parse policy number" << endl;
    }
    
    if (strcmp(args[i],"--behavior-number") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> behavior_policy_number))
        cout << "Could not parse policy number" << endl;
    }
    
    if (strcmp(args[i],"--cee-mrp") == 0) {
      cee_mrp = true;
    }
    
    if (strcmp(args[i],"--cee-mdp") == 0) {
      cee_mdp = true;
    }
    
    if (strcmp(args[i],"--mdp") == 0) {
      mdp = true;
    }
    if (strcmp(args[i],"--cee-exp-sarsa-mdp") == 0) {
      cee_exp_sarsa_mdp = true;
    }
    if (strcmp(args[i],"--cee-sarsa-mdp") == 0) {
      cee_sarsa_mdp = true;
    }
    if (strcmp(args[i],"--cee-psec-sarsa-mdp") == 0) {
      cee_psec_sarsa_mdp = true;
    }   
    if (strcmp(args[i],"--ris") == 0) {
      ris = true;
    }

    if (strcmp(args[i], "--deterministic-prob") == 0) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> deterministic_prob))
        cout << "Could not parse deterministic prob" << endl;
    }
    if (strcmp(args[i], "--alpha") == 0) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> alpha))
        cout << "Could not parse learning rate" << endl;
    }
    if (strcmp(args[i],"--lstd") == 0) {
      lstd = true;
    }
    if (strcmp(args[i],"--rho-new-est") == 0) {
      rho_new_est = true;
    }
    if (strcmp(args[i],"--rho-new-est") == 0) {
      rho_new_est = true;
    }
    if (strcmp(args[i],"--state-action") == 0) {
      state_action = true;
    }
    if (strcmp(args[i],"--exp-sarsa") == 0) {
      exp_sarsa = true;
    }
  }

  // always considering GridWorld
  //int domain = GRIDWORLD_NON_TRAJ_LIM;//GRIDWORLD_TH;
  int domain = STOCHASTIC_GRIDWORLD_NON_TRAJ_LIM;
  Environment *env = getDomain(domain, deterministic_prob);
  
  // number of times to process batch
  // used for VF learning in the worst case if not auto convergence
  int batch_process_steps = 100000;
  string method_name;
  string dp_type;
  if (behavior_policy_number == -1)
    behavior_policy_number = policy_number;
  
  int method; 
  if (ris) {
    method = 0;
    method_name = "BatchRISTD";
  } else {
    method = 1;
    method_name = "BatchTD";
  }

  if (mdp) {
    dp_type = "mdp";
  } else if(cee_mrp) {
    dp_type = "cee-mrp";
  } else if (cee_mdp) {
    dp_type = "cee-mdp";
  } else if(cee_exp_sarsa_mdp){
    dp_type = "cee-exp-sarsa-mdp";
  } else if(cee_sarsa_mdp){
    dp_type = "cee-sarsa-mdp";
  } else if(cee_psec_sarsa_mdp){
    dp_type = "cee-psec-sarsa-mdp";
  }

  printf("================= SINGLE RUN =================\n");
  printf("Method name: %s\n", method_name.c_str());    
  printf("RIS: %d\n", ris);
  printf("state-action: %d\n", state_action);
  printf("expected sarsa: %d\n", exp_sarsa);
  printf("Seed (trial number): %d\n", seed);    
  printf("Number of trajs: %d\n", num_trajs);    
  printf("Learning rate: %f\n", alpha);    
  printf("On policy %d (eval policy: %d and behavior policy: %d)\n", policy_number == behavior_policy_number, policy_number, behavior_policy_number);
  printf("Computing certainty-equivalence Markov reward process: %d\n", cee_mrp);
  printf("Computing certainty-equivalence Markov decision process: %d\n", cee_mdp);
  printf("Computing true Markov reward process: %d\n", mdp);
  printf("DP Method: %s\n", dp_type.c_str());

  //string outfile_name("results/vf/" + method_name + "_trajs_" + std::to_string(num_trajs) + "_trial_" + std::to_string(i) + ".pb");

  if (num_trajs == -1) {

    int start_trajs[] = {10, 200};//, 2000};
    int traj_increments[] = {10, 100};//, 1000};
    int end_trajs[] = {100, 1000};//, 10000};

    for (int i = 1; i >= 0; i--) {
        int start = start_trajs[i];
        int increment = traj_increments[i];
        int end = end_trajs[i];
        for (int m = start; m <= end; m += increment) {

            string temp_name = outfile_name + std::to_string(m);
            printf("new outfile name: %s\n", temp_name.c_str());    
            printf("traj: %d\n", m);
            BatchEvaluate(env,
                        policy_number,
                        behavior_policy_number,
                        m,
                        seed,
                        temp_name,
                        method,
                        method_name,
                        print_freq,
                        batch_process_steps,
                        dp_type,
                        deterministic_prob,
                        alpha,
                        rho_new_est);
        }
    }
  } else {
      if (lstd) {
          LSTDBatchEvaluate(env,
                        policy_number,
                        behavior_policy_number,
                        num_trajs,
                        seed,
                        outfile_name,
                        method,
                        method_name,
                        print_freq,
                        batch_process_steps,
                        dp_type,
                        deterministic_prob,
                        alpha);
      } else if (state_action){
          BatchEvaluateSA(env,
                        policy_number,
                        behavior_policy_number,
                        num_trajs,
                        seed,
                        outfile_name,
                        method,
                        method_name,
                        ris,
                        print_freq,
                        batch_process_steps,
                        dp_type,
                        deterministic_prob,
                        alpha,
                        rho_new_est,
                        exp_sarsa);

      }

      else {
          BatchEvaluate(env,
                        policy_number,
                        behavior_policy_number,
                        num_trajs,
                        seed,
                        outfile_name,
                        method,
                        method_name,
                        print_freq,
                        batch_process_steps,
                        dp_type,
                        deterministic_prob,
                        alpha,
                        rho_new_est);
        }
    }
  
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}

