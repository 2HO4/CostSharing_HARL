

<p align="center"><img src="resources/logo/uva.png" alt="drawing" width="200"/></p>


# **<p align="center">Global Warming Mitigation: An Empirical Multi-Agent Reinforcement Learning Approach</p>**

*<p align="justify">This project is part of a Bachelor's thesis for the Econometrics and Data Science programme, which aims to investigate the efficacy of serial cost-sharing taxation policies in regulating emission levels from petroleum industry production using the advanced [Heterogeneous-Agent Reinforcement Learning (HARL)](https://github.com/PKU-MARL/HARL) model. The goal of this project is to explore how different serial cost-sharing taxation policies can regulate emission levels in the petroleum industry. By leveraging a multi-agent reinforcement learning approach, the study identifies key factors influencing firms’ economic viability while achieving environmental targets. The thesis document can be accessed in the `docs` folder.</p>*


## Table of Contents
- [Running the Project](#running-the-project)
  - [Google Colab](#google-colab)
  - [Local Machine / Container](#local-machine)
    1. [Prerequisites](#1-prerequisites)
    2. [Clone the Repository and Prepare the Project Environment](#2-clone-the-repository-and-prepare-the-project-environment)
    3. [Train the Model](#3-train-the-model)
    4. [Inspect the Ongoing Results](#4-inspect-the-ongoing-results)

- [Extending the Project](#extending-the-project)
  - [Adding Custom Cost Functions](#adding-custom-cost-functions)
  - [Adding Custom Cost-Sharing Algorithms](#adding-custom-cost-sharing-algorithms)

- [Contributing](#contributing)

- [License](#license)


## Running the Project
Follow the steps below to set up and run the project either using Google Colab, on a local machine, or a virtual container.

### Google Colab
The simplest way to execute the project is through Google Colab. You can run the project by accessing the [following Google Colab notebook](https://colab.research.google.com/github/2HO4/CostSharing_HARL/blob/main/process_GoogleColab.ipynb).

### Local Machine / Container
On the other hand, to run the project on your local machine, you can also run the notebook `process.py` which completes all the following manual steps as explained below. You can also follow these steps instead of using a .ipynb notebook which gives you better understanding of the project and autonomy in execution.

#### **1. Prerequisites**
To run this project on your local machine or container, ensure you have [Python 3.8(.19)](https://www.python.org/downloads/release/python-3819/) and [Git](https://git-scm.com/downloads) installed. 

Optionally, if you want to have a self-contained environment for the project instead of having all required package installed on your machine's main pip, you can either use a virtual environment (e.g., [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)) or a virtual container (e.g., [Docker](https://docs.docker.com/engine/install/), [Podman](https://podman.io/docs/installation)).

#### **2. Clone the Repository and Prepare the Project Environment**
```bash
# Clone the repositoary
git clone https://github.com/2HO4/CostSharing_HARL.git
cd CostSharing_HARL

# Prepare the project
python utilities/prepare_project.py  # for CPU cores
# or
python utilities/prepare_project.py --gpu --fix_nv  # for GPU cores
```

#### **3. Train the Model**
The model can be trained using one of nine advanced multi-agent reinforcement learning algorithm on the cost-sharing environment. The training can be executed by running `train.py` with the following usage syntax:
```bash
python train.py [-h] --algo {happo,hatrpo,haa2c,haddpg,hatd3,hasac,had3qn,maddpg,matd3,mappo,custom} --env {default,custom} [--exp_name EXP_NAME] [--load_config LOAD_CONFIG] [*args]

Required arguments:
--algo {happo,hatrpo,haa2c,haddpg,hatd3,hasac,had3qn,maddpg,matd3,mappo,custom}
    Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo, or custom. (default: happo)
--env {default,custom}
    Environment type. Choose from: default or custom. (default: default)

Optional arguments:
-h, --help
    Show this help message and exit
--exp_name EXP_NAME
    Experiment name. (default: installtest)
--load_config LOAD_CONFIG
    If set, load existing experiment config file instead of reading from yaml config file. (default: )
*args
    Any other parameters to be used as part of either the algorithm or environment configuration.
```
The configurations for each individual learning algorithms can be modified in `HARL/harl/configs/algos_cfgs`. On the other hand, instead of editing each configuration manually and then change the training command, you can edit the custom algorithm configuration which works for all available MARL algorithms and only use `--algo custom`.

Similarly, you can also modify all parameters of the environment by editing the custom environment configuration and use `--env custom`.

The configurations for the custom algorithm and environment are specified in `resources/configs/algorithm.yaml` and `resources/configs/environment.yaml`, respectively.

The cost-sharing environment can be customized through these parameters:
- **`details_firms`**: Details of the firms participating in the game.
- **`date_start`**: Start time of the game.
- **`date_end`**: End time of the game.
- **`emissions_max`**: Maximum emissions allowed in the game.
- **`seed`**: Seed for random number generation.
- **`price_is_constant`**: Flag indicating whether the price of gasoline is constant. If it is a float, then it is the 'constant' price of gasoline.
- **`price_ceiling`**: The maximum price of gasoline in the game.
- **`price_floor`**: The minimum price of gasoline in the game.
- **`discount_demand`**: The discount rate of the remaining unsatisfied demand. For example, if it is 0.25, then 1/4 of the remaining demand will be carried to the next period.
- **`portion_cost`**: The value that manipulates the scaling of the total cost of producing gasoline.
- **`portion_reward`**: The portion of a firm's earnings that becomes the reward.
- **`portion_punishment`**: The portion of total earnings that is taken as a punishment when the demand is not satisfied.
- **`reward_final`**: The reward given to a firm when it ends the game.
- **`quota_production`**: The number of barrels of gasoline that a single firm needs to produce in a week.
- **cap_production**: The maximum number of barrels of gasoline that a single firm can produce in a week.
- **`hide_others_moves`**: Flag indicating whether past production of firms should be hidden from each firm.
- **`cost_function`**: Cost function of the number of barrels produced which is used in the game.
- **`algorithm`**: Cost-sharing algorithm used in the game.
- **`render_mode`**: Render mode for visualization.

For instance, without any modification to the project, you can run this command to begin training:
```bash
python train.py --algo custom --env custom --exp_name oligopoly_firms4
```

#### **4. Inspect the Ongoing Results**
Inspect the ongoing results of the current model. Make sure the model directory is not empty before running this step. Replace `<DATE>` with the appropriate date of your model directory. For instance:
```bash
python train.py --algo custom --env custom --exp_name example_oligopoly_firms4 --use_render True --model_dir results/cost_sharing/cost_sharing/happo/oligopoly_firms4/seed-<DATE>/models
```


## Extending the Project

### Adding Custom Cost Functions
You can implement additional cost functions by adding a function `cost_<NAME>(n_barrels_supplied: int)` in the file `resources/economics/cost_functions.py`.

### Adding Custom Cost-Sharing Algorithms
You can implement additional cost-sharing algorithms by adding a function `cost_sharing_<NAME>(cost_function: Callable[[int], float], quantities_to_produce: dict)` in the file `resources/economics/cost_sharing_algorithms.py`.


## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

For more details, refer to the thesis document in the `docs` folder. If you have any questions or issues, feel free to open an issue on GitHub.
