## Paper Title: Enhancing Collective Communication in MCM Accelerator for Deep Learning Training

### Directory Structure
1. HPCA_2024_final: Contains the results of all the simulation. This results can be used to reproduce the graphs presented in the paper.
2. src: Contains al the source code
   - allreduce: This subdirectory contains all the AllReduce algortims presented in the paper.
   - booksim2: BookSim2 simulator source code
   - SavedTrees: As MultiTree takes long time to generate trees when number of accelerators are large, we initially build the tree and load during simulation.
   - SCALE-Sim: Stores code for scale-sim
3. utils: Contains python and shell scripts to reproduce the results.
   - python_scripts: Contains python scripts to reproduce graphs in the paper. It uses results from HPCA_2024_final directory.
   - run_scripts: Contains shell scripts to run the simulations.

### Software Requirements
1. Ubuntu 22.04(Other Ubuntu version should work as well)
2. Python 3.7.15
3. ScaleSim + BookSim(We have included both the simulators with this repository)

### Setup
Create a new virtual environment

`python3 -m venv venv`

Activate the virtual environment

`source venv/bin/activate`

Install the required dependencies from requirements.txt file (If required upgrade the pip)

`pip install -r requirements.txt`

Run the following commands to finalize the environment setup.

`source setup_env.sh`

The above command should show "source successfully" message.

`cd src`

`cd booksim2`

`cd src`

`make clean` 

`make lib`

It binds booksim with python code. Now go back to the root folder.

Run the following commands to update config files topology directory.

`cd src`

`python update_cfg_files.py`

It will update the config file by appending the root directory of your system.
You need to run this just once. After that you can reuse the updated config files for your experiment. Then go back to root folder.

### Reproduce the results of the paper

We are providing all the simulation output files in **HPCA_2024_final** folder.
To reproduce the graphs from the papers, first go to the **python_scripts** folder

`cd utils/python_scripts`

1. To reproduce the results for Figure 8, run the following command. All the results for bandwidth are in **HPCA_2024_final/bandwidth** folder. Running the following script will generate a file named *bandwidth.pdf*.
   - `python plot_bandwidth.py`
2. To reproduce the results for Figure 9, run the following command. All the results for scalability are in **HPCA_2024_final/scalability** folder. Running the following script will generate a file named *scalability.pdf*.
   - `python plot_scalability.py`
3. To reproduce the results for Figure 10, run the following command. All the results for real-world DNN models are in **HPCA_2024_final/models** folder. To get the training time of DNN models, we simulate the models with ScaleSim simulator.
   - `python plot_model_even.py` will generate models_even.pdf file(Fig 10 left)
   - `python plot_model_odd.py` will generate models_odd.pdf file(Fig 10 right)
4. To reproduce the results for Figure 11, run the following command. All the results for overlapping computation and communication are in **HPCA_2024_final/overlap** folder. Running the following script will generate a file named *overlap.pdf*.
   - `python plot_overlap.py`
5. To reproduce the results for Figure 12, run the following command. All the results for link utilization are in **HPCA_2024_final/utilization** folder. Running the following script will generate a file named *link_utilization.pdf*. This run may take hours to finish.
   - `python plot_link_utilization.py`
6. To reproduce the results for Figure 13, run the following command. All the results for simba results are in **HPCA_2024_final/simba** folder.
   - `python plot_simba_results_32.py` will generate models_simba_32.pdf file(Fig 13 left)
   - `python plot_simba_results_16.py` will generate models_simba_16.pdf file(Fig 13 right)
7. To reproduce the results for Figure 14, run the following command. All the results for best chunk size are in **HPCA_2024_final/chunk_utilization** folder. Running the following script will generate a file named *chunk.pdf*.
   - `python plot_chunk_utilization.py`
    
Below we present the name used in the code of the algorithms presented in paper.
1. TTO('mesh_overlap_2d_1')
2. Ring AllReduce Algorithm with Even-sized Mesh('ring')
3. Ring AllReduce Algorithm with Odd-sized Mesh('ring_odd')
4. Hierarchical Two-dimensional Ring AllReduce Algorithm('ring2dn')
5. Double Binary Tree('dtree')
6. Multitree('multitree')
7. Bidirectional Ring AllReduce Algorithm with Even-sized Mesh('ring_bi')
8. Bidirectional Ring AllReduce Algorithm with Odd-sized Mesh('ring_odd_bi')


## Reproduce the output results

We provide the scripts to reproduce all the results in **utils/run_scripts** folder.
Note that, most of the scripts parallely run multiple simulations simultaneously.
So, running them in a small core machine may take most of the resources. Running the
scripts will create a folder named **HPCA2024** folder in the root directory and all
the outputs will be stored there. Each simulation will produce a log file and a json
file where json file contains the outputs. Number of cycles for the simulation can
be found in results/performance inside the json file.

1. Scripts in **run_scripts/bw/*.sh** will reproduce results of bandwidth utilization.
2. Scripts in **run_scripts/models/**.sh* will reproduce results of all DNN benchmark performances.
3. Scripts in **run_scripts/overlap/**.sh* will reproduce results of communication and computation overlap.
4. Scripts in **run_scripts/scalability/**.sh* will reproduce results of scalability.
5. Scripts in **run_scripts/simba/**.sh* will reproduce results of all DNN benchmark performances.
6. Scripts in **run_scripts/chunk_sensitivity.sh** will reproduce results of chunk sensitivity of TTO.
7. Scripts in **run_scripts/link_utilization.sh** will reproduce results of link utilization.


To test TTO with AlphaGoZero, you can run the following command.
`./utils/mesh_overlap_2d_1_models_test.sh`

This will create a json file same as the json file stored in **test_result** folder. All other
scripts in **utils/run_scripts** folder should create json file similar to the outputs provided in **HPCA_2024_final** folder.

We use the following pattern for naming the output files.
`{exp_type}_{model_name/data_size}_{allreduce_alg_type}_{total_nodes}_mesh.json`

So, to test bw utilization of TTO with 512 MB data and 16 nodes, the output file name will be **bw_134217728_mesh_overlap_2d_1_16_mesh.json**

Here is the number of experiments in each file.
- bw/ring_bi_odd_bw.sh: 22
- bw/tree_dtree_bw.sh: 44
- bw/ring_uni_even_bw.sh: 22
- bw/ring_uni_odd_bw.sh: 22
- bw/mesh_overlap_2d_1_bw.sh: 44
- bw/ring_bi_even_bw.sh: 22
- bw/ring_2dn_bw.sh: 44
- bw/multitree_bw.sh: 44
- models/ring_uni_even_models.sh: 7
- models/tree_dtree_models.sh: 14
- models/ring_2dn_models.sh: 14
- models/ring_bi_odd_models.sh: 7
- models/mesh_overlap_2d_1_models.sh: 14
- models/multitree_models.sh: 14
- models/ring_uni_odd_models.sh: 7
- models/ring_bi_even_models.sh: 7
- overlap/comm_comp_overlap.sh: Around 1002
- overlap/training_overlap.sh: 7
- scalability/ring_bi_odd_scalability.sh: 7
- scalability/ring_uni_odd_scalability.sh: 7
- scalability/ring_bi_even_scalability.sh: 7
- scalability/mesh_overlap_2d_1_scalability.sh: 14
- scalability/multitree_scalability.sh: 14
- scalability/ring_2dn_scalability.sh: 116
- scalability/ring_uni_even_scalability.sh: 7
- simba/simba_exp.sh: 84
- simba/simba_train.sh: 14
- chunk_sensitivity.sh: 11
- link_utilization.sh: 5