# script to be run at the start of the project to prepare the project of multi-agent reinforcement learning for cost-sharing mechanism

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the project for multi-agent reinforcement learning for cost-sharing mechanism', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--gpu', 
        action='store_true', 
        help='Install required packages for GPU support'
    )

    parser.add_argument(
        '--fix_nv', 
        action='store_true', 
        help='Fix Pytorch DLL for Nvidia GPU'
    )

    parser.add_argument(
        '--venv', 
        type=str, 
        default='pip', 
        choices=['pip', 'conda', 'mamba', 'virtualenv'], 
        help='Specify the virtual environment to use'
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    # Install required packages for the project
    if args['venv'] in ['conda', 'mamba']:
        subprocess.run([args['venv'], 'env', 'create', '-f', 'environment.yml', '-p', '.venv'])
        subprocess.run([args['venv'], 'activate', '.venv'])
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    elif args['venv'] == 'virtualenv':
        subprocess.run(['pip', 'install', 'virtualenv'])
        subprocess.run(['virtualenv', '.venv'])
        subprocess.run(['source', '.venv/bin/activate'])
        subprocess.run(['pip', 'install', 'python==3.8'])
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    else:
        subprocess.run(['pip', 'install', 'python==3.8'])
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    
    # Install required packages for GPU support
    if args['gpu']:
        subprocess.Popen(['pip' 'install', 'torch==2.3.0', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
    
    # Fix Pytorch DLL for Nvidia GPU
    if args['fix_nv']:
        subprocess.Popen(['python', 'fix_Nvidia.py', '--input', 'torch*.dll', '--backup'])

    # Clone the required HARL repository
    subprocess.run(['git', 'clone', 'https://github.com/PKU-MARL/HARL.git'])
    subprocess.call('cd HARL', shell=True)
    subprocess.run(['pip', 'install', '-e', '.'])
    subprocess.call('cd ..', shell=True)

    # Move the modified files to the HARL repository
    subprocess.run(['python', 'move_modified_files.py'])


if __name__ == '__main__':
    main()
