**Robust RMAB (RMABDO) and Deep RL for RMAB (RMABPPO, MA-RMABPPO)**
==================================
Robust Restless Bandits: Tackling Interval Uncertainty with Deep Reinforcement Learning

Code that accompanies paper: Killian JA, Xu L, Biswas A, Tambe M. Robust Restless Bandits: Tackling Interval Uncertainty with Deep Reinforcement Learning. Arxiv.


## Setup

Main file for RMABDO, the algorithm for Robust Multi-action RMAB is `double_oracle.py`

Main file for RMABPPO, the algorithm for Deep RL for binary and multi-action RMAB is `agent_oracle.py`

Main file for MA-RMABPPO, the algorithm for Multi-agent Deep RL for binary and multi-action RMAB is `nature_oracle.py`

#### To install follow these directions (generic version):

**Note:** These are generic steps to install what's needed to run the code. However, please see the bottom of the readme for verified/reproducible setup instructions that start from a new digital ocean linux server.

- Clone the repo:
- `git clone https://github.com/killian-34/RobustRMAB.git`
- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

If you have any issues with mpi4py or tkinter, try these steps (on linux)
- `apt-get update`
- `sudo apt install python3-mpi4py`
- `sudo apt-get install python3-tk`

Then run `pip3 install -e .` again once these successfully install.


To test that the install worked correctly for RMABPPO, run:

`bash run/job.run_rmabppo_tiny.sh`

To test that the install worked correctly for RMABDO, run:

`bash run/job.run_do_tiny.sh`




## Running Code 

### Running RMABDO - Double Oracle for Robust Multi-action RMAB

To run a basic example of the double oracle framework using the `counterexample` domain from the paper, run 
`bash run/job.run_do_counterexample.sh`

The parameter settings for RMABDO and the underlying algorithms for the Agent Oracle (RMABPPO) and Nature Oracle (MA-RMABPPO) are set in the corresponding file `job.run_do_counterexample.sh`. More details on the parameters can be found by running 

`python3 double_oracle.py --help`


Note that this runs the code without the hawkins baselines since these require installation of gurobi and gurobipy. 
To run the baselines, first install gurobi and gurobipy (you can obtain a free academic license). Detailed instructions on how to install these are given at the bottom of the readme in the "**Setup steps, Digital Ocean install**" section.

then in `run/job.run_do_tiny.sh` or `run/job.run_do_counterexample.sh` change `no_hawkins=1` to `no_hawkins=0` to run the hawkins baselines.


### Running RMABDO - Double Oracle for Robust Multi-action RMAB

To run a basic example of RMABPPO using the `counterexample` domain from the paper, run 
`bash run/job.run_rmabppo_counterexample.sh`

The parameter settings for RMABPPO are set in the corresponding file `job.run_rmabppo_counterexample.sh`. More details on the parameters can be found by running 

`python3 agent_oracle.py --help`


Again, this runs the code without the hawkins baseline since it requires installation of gurobi and gurobipy. 
To run the baselines, first install gurobi and gurobipy (you can obtain a free academic license). Detailed instructions on how to install these are given at the bottom of the readme in the "**Setup steps, Digital Ocean install**" section.

then in `run/job.run_rmabppo_tiny.sh` or `run/job.run_rmabppo_counterexample.sh` change `no_hawkins=1` to `no_hawkins=0` to run the hawkins baseline.


## Notes:
- The implementation of RMABPPO (Agent Oracle) can be found in `robust_rmab/algos/rmabppo/rmabppo_core.py`
- The implementation of MA-RMABPPO (Nature Oracle) can be found in `robust_rmab/algos/ma_rmabppo/ma_rmabppo_core.py`
- The double oracel portion of the codebase is currently not setup to allow multi processing from the double_oracle.py script. However, the underyling RL code was built on the SpinningUp repository which supports MPI and so the code is entangled with MPI-based function calls, meaning MPI must still be installed to run double_oracle.py. This will be addressed in future iterations.
- However, standalone RMABPPO is setup for multi-processing and can be controlled with the `--cpu` option in `agent_oracle.py`



## Setup steps, Digital Ocean install
Tested as of July 3, 2021

- Create a droplet on Digital Ocean using the following specifications
- `4 GB Memory / 2 Intel vCPUs / 80 GB Disk / NYC1 - Ubuntu 20.04 (LTS) x64`
- After hitting Create Droplet, give the server a minute to spin up, then go to your terminal and ssh into the server using the ipv4 address that Digital Ocean assigned to the droplet, e.g., `ssh -Y root@000.000.000.00`

Once connecte to the server, run the following commands:
- `git clone https://github.com/killian-34/RobustRMAB.git`
- `cd RobustRMAB`
- Create directory structure: `bash make_dirs.sh`
- `apt-get update`
- `apt install python3-pip --fix-missing`
- `sudo apt install python3-mpi4py`
- `sudo apt-get install python3-tk`
- `pip install -e .`
- Run this command to test RMABPPO: `bash run/job.run_rmabppo_tiny.sh`
- Run this command to test RMABDO: `bash run/job.run_do_tiny.sh`


#### If you want to run Hawkins policies, e.g., `no_hawkins=0`, you need to install gurobi and gurobipy by following these steps
- In a web browser Register for a Gurobi account or login at https://www.gurobi.com/downloads/end-user-license-agreement-academic/ 
- Navigate to https://www.gurobi.com/downloads/ and select `Gurobi Optimizer`
- Review the EULA, then click `I accept the End User License Agreement`
- Identify the latest installation... as of writing, it is 9.1.2, and the following commands will reflect that. However, if the latest version has changed, you can replace 9.1.2 in the following commands with the newer version number and/or links on the Gurobi website.
- Navigate to `https://www.gurobi.com/wp-content/uploads/2021/04/README_9.1.2.txt` and read the README
- Back on the digial ocean server terminal: `mkdir tools`
- `cd tools`
- `wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz`
- `tar xvzf gurobi9.1.2_linux64.tar.gz`
- Add the following lines to your ~/.bashrc file, e.g., via `vim ~/.bashrc`
```
export GUROBI_HOME="/root/tools/gurobi912/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
``` 
- Run `source ~/.bashrc`
- On the browser, navigate to https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- Review the conditions, then click `I accept these conditions`
- Scroll down to **Installation** and copy the command that looks like `grbgetkey 00000000-0000-0000-0000-000000000000`, then paste and run in the server terminal window
- Enter `Y` to select the default options when prompted
- `cd ~/RobustRMAB/`
- `pip install gurobipy`
- `vim run/job.run_do_tiny.sh`
- Change `no_hawkins=1` to `no_hawkins=0`
- `bash run/job.run_do_tiny.sh`
- `vim run/job.run_rmabppo_tiny.sh`
- Change `no_hawkins=1` to `no_hawkins=0`
- `bash run/job.run_rmabppo_tiny.sh`


