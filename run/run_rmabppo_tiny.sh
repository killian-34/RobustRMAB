python3 ${1}/agent_oracle.py --hid 16 -l 2 --gamma 0.9 --cpu 1 \
--exp_name ${5} \
--home_dir ${1} \
-s ${2} \
--cannon ${3} \
--data ${4} \
--save_string ${5} \
-N ${6} -B ${7} \
\
--agent_steps 10 \
--agent_epochs ${9} \
--agent_init_lambda_trains 0 \
--agent_clip_ratio 2 \
--agent_final_train_lambdas 2 \
--agent_start_entropy_coeff 0.5 \
--agent_end_entropy_coeff 0 \
--agent_pi_lr 2e-3 \
--agent_vf_lr 2e-3 \
--agent_lm_lr 2e-3 \
--agent_train_pi_iters 20 \
--agent_train_vf_iters 20 \
--agent_lamb_update_freq 4 \
--robust_keyword ${8} \

exp_name=${5}_n${6}b${7}d${4}r${8}p0
python3 ${1}/robust_rmab/simulator.py --discount 0.9 \
--budget ${7} \
 --data ${4} \
-N ${6} \
-s ${2} -ws ${2} \
-rlmfr ${1}/data/${exp_name}/${exp_name}_s${2}/ \
-L 10 \
 -n 3 \
 --robust_keyword ${8} \
 --file_root ${1} \
 --save_string ${5} \
 --no_hawkins ${10}