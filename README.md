# Investigation of Cost Sharing Policies Through Multi-Agent Reinforcement Learning


## Introduction
This project aims to investigate the effectiveness of cost sharing policies using multi-agent reinforcement learning techniques. 

## Installation
To run this project, you need to have Python 3.8 installed. Clone the repository and install the required dependencies using the following commands:
- If `conda` environment is used:
```bash
conda create --path .venv --file environment.yml
conda activate ./.venv
pip install -r requirements.txt

```

- If `pip` environment is used:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

To add an environment:
HARL\harl\utils\envs_tools.py

To add the environment's logger:
HARL\harl\envs\__init__.py

To register name of the environment/task:
HARL\harl\utils\configs_tools.py

To register number of agents:
HARL\harl\utils\envs_tools.py

To add an algorithm:
HARL\harl\runners\__init__.py
HARL\harl\algorithms\actors\__init__.py




To train with Pytorch using GPU:
pip3 install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

If this error occurs:
OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading ".../.venv\lib\site-packages\torch\lib\cudnn_cnn_infer64_8.dll" or one of its dependencies.
- run:
python fix_nvidia.py --input=.venv\lib\site-packages\torch\lib\*.dll


Additional cost-sharing algorithms can be implement as a function `cost_sharing_<NAME>(cost_function: Callable[[int], float], quantities_to_produce: dict)` in the file `resources/economics/cost_sharing_algorithms.py` 
