
#include "Environment.hpp"
#include "batch_methods.h"

void BatchEvaluate(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs, 
                       int seed, string outfile, int method,
                       string method_name,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha,
                       bool rho_new_est);
void BatchEvaluateSA(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs, 
                       int seed, string outfile, int method,
                       string method_name, bool ris,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha,
                       bool rho_new_est, bool exp_sarsa);
void LSTDBatchEvaluate(Environment *env, int target_policy_number,
                       int behavior_policy_number, int num_trajs,
                       int seed, string outfile, int method, string method_name,
                       int print_freq, int batch_process_steps,
                       string dp_type, float deterministic_prob, double alpha);

// TODO: temporary sol, change it!
double compute_msve_batch_sa(BatchVFMethod* batch_vf_method, Model m, int max_time, int num_states, int num_actions, string dp_type);
double compute_msve_batch(BatchVFMethod* batch_vf_method, Model m, int max_time, int num_states, string dp_type);
double compute_msve_LSTD(VectorXd w, Model m, int num_states, string dp_type);


