Instructions to run Gridworld Experiments

1. Create obj/ in current directory
2. Compile protobuf files
2.1. cd protos; bash compile.sh
2.2. return to root dir: cd ..
3. compile code: make -j8
3.1. Sometimes linking fails. If you re-run make a couple times it should resolve.

Content:
Code reproduces Gridworld related graphs: Figure 1, 3, 4

mkdir results

==== Figure 1 ===
Figure 1(a)
mkdir results/td
python3 run_gw.py results/td/ --num_trials 50 --predefined_batches --fig_num 1 --on-policy
python3 scripts/vf_param_sweep_plot.py results/td/
view file param_sweep_result.pdf

Figure 1(b)
mkdir results/td_off
python3 run_gw.py results/td_off --num_trials 50 --predefined_batches --fig_num 1
python3 scripts/vf_param_sweep_plot.py results/td_off/
view file param_sweep_result.pdf

==== Figure 3 ===
Figure 3(a)
mkdir results/lstd
python3 run_gw.py results/lstd/ --num_trials 50 --predefined_batches --fig_num 3 --on-policy
python3 scripts/vf_param_sweep_plot.py results/lstd/
view file param_sweep_result.pdf

Figure 3(b)
mkdir results/lstd_off
python3 run_gw.py results/lstd_off --num_trials 50 --predefined_batches --fig_num 3 
python3 scripts/vf_param_sweep_plot.py results/lstd_off/
view file param_sweep_result.pdf

==== Figure 4 ===
Figure 4(a)
mkdir results/cee
python3 run_gw.py results/cee --num_trials 50 --predefined_batches --fig_num 41
python3 scripts/vf_plot.py results/cee/
view file PSEC-CEE.pdf

Figure 4(b)
mkdir results/unvisited
python3 run_gw.py results/unvisited/ --num_trials 50 --predefined_batches --fig_num 42
python3 scripts/vf_plot.py results/unvisisted
view file unvisited_fraction

Figure 4(c)
mkdir results/stoch
python3 run_gw.py results/stoch/ --num_trials 100 --predefined_batches --fig_num 43
python3 scripts/stochasticity_plot.py results/stoch/
view file stoch_plot.pdf 

