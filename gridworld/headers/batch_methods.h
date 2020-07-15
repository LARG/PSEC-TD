
#ifndef _BATCH_METHODS_
#define _BATCH_METHODS_

#include "Policy.h"
#include <vector>
#include <Eigen/Dense>
#include "Trajectory.h"
#include <limits>

class BatchVFMethod {

public:
	BatchVFMethod() {}
	virtual void update() = 0;
    virtual bool accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) = 0;
    virtual void flush_accumulation() = 0;
	virtual void compute_RIS_estimates(vector<Trajectory> all_data) = 0;
	virtual bool has_converged(bool td_error_check, double eps) = 0;
    virtual void compute_traj_RIS_IWs(vector<Trajectory> & all_data, const Policy &eval_pi) = 0;
    virtual void update_learning_rate(double new_lr) = 0;
    Eigen::MatrixXd v_;
    Eigen::MatrixXd v_old_;
    Eigen::MatrixXd v_accum_;
    vector<MatrixXd> Q_; // [s][a]
    vector<MatrixXd> Q_old_; // [s][a]
    vector<MatrixXd> Q_accum_; // [s][a]   

    double max_td_error = std::numeric_limits<double>::min();

protected:

    int num_states_;
	int max_traj_length_;
	float lambda_;
	double alpha_;
    bool rho_new_est_;
	Eigen::MatrixXd e_;
	Eigen::MatrixXd v_grad_;

};

class BatchTD : public BatchVFMethod {

public:
	BatchTD(int num_states, int max_traj_len, float lambda, float alpha, bool rho_new_est);
	virtual void update() override;
	virtual bool accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) override;
    virtual void flush_accumulation() override;
    virtual void compute_RIS_estimates(vector<Trajectory> all_data) override;
	virtual bool has_converged(bool td_error_check, double eps) override;
    virtual void compute_traj_RIS_IWs(vector<Trajectory> & all_data, const Policy &eval_pi) override;
    virtual void update_learning_rate(double new_lr) override;
};

class BatchRISTD : public BatchVFMethod {

public:
	BatchRISTD(Policy pi, int num_states, int num_actions, int max_traj_len, float lambda, float alpha, bool rho_new_est);
	virtual void update() override;
	virtual bool accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) override;
    virtual void flush_accumulation() override;
    virtual void compute_RIS_estimates(vector<Trajectory> all_data) override;
	virtual bool has_converged(bool td_error_check, double eps) override;
    virtual void compute_traj_RIS_IWs(vector<Trajectory> & all_data, const Policy &eval) override;
    virtual void update_learning_rate(double new_lr) override;

private:
    vector<Eigen::MatrixXd> state_action_counts_;
    vector<Eigen::VectorXd> state_counts_;
    vector<vector<MatrixXd>> s_a_s_counts_;
    vector<vector<vector<MatrixXd>>> s_a_s_a_counts_;
    Policy policy_;
};

class BatchSARSA : public BatchVFMethod {

public:
	BatchSARSA(Policy pi, int num_states, int num_actions, int max_traj_len, float lambda, float alpha, bool expected_sarsa, bool ris, bool rho_new_est);
	virtual void update() override;
	virtual bool accumulate(int t, int s, int a, double R, int sPrime, int aPrime, double IW) override;
    virtual void flush_accumulation() override;
    virtual void compute_RIS_estimates(vector<Trajectory> all_data) override;
	virtual bool has_converged(bool td_error_check, double eps) override;
    virtual void compute_traj_RIS_IWs(vector<Trajectory> & all_data, const Policy &eval) override;
    virtual void update_learning_rate(double new_lr) override;

private:
    Policy policy_;
    bool exp_sarsa_;
    bool ris_;
    bool rho_new_est_;
    int num_actions_;
    vector<Eigen::MatrixXd> state_action_counts_;
    vector<Eigen::VectorXd> state_counts_;
    vector<vector<MatrixXd>> s_a_s_counts_;
    vector<vector<vector<MatrixXd>>> s_a_s_a_counts_;
    vector<VectorXd> actionProbabilities; // [s][a]

};
#endif
