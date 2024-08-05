# script to be run at the start of the project to prepare the project of multi-agent reinforcement learning for cost-sharing mechanism

import argparse
import subprocess
import os


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
    args = vars(args)  # convert to dict

    def is_GoogleColab():
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    # Install required packages for the project
    if is_GoogleColab():
        subprocess.run(['pip', 'install', 'numpy==1.23.5', 'pettingzoo==1.22.2', 'supersuit==3.7.0', 'pefile==2023.2.7'])

    elif args['venv'] in ['conda', 'mamba']:
        subprocess.run([args['venv'], 'env', 'create', '-f', 'environment.yml', '-p', '.venv'])
        subprocess.run([args['venv'], 'activate', './.venv'])
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

    elif args['venv'] == 'virtualenv':
        subprocess.run(['pip', 'install', 'virtualenv'])
        subprocess.run(['virtualenv', '.venv'])
        subprocess.run(['source', '.venv/bin/activate'])
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
        
    else:
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    
    # Install required packages for GPU support
    if args['gpu']:
        subprocess.Popen(['pip', 'install', 'torch==2.3.0', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
    
    # Fix Pytorch DLL for Nvidia GPU
    if args['fix_nv']:
        subprocess.Popen(['python', 'utilities/fix_nvidia.py', '--input', 'torch*.dll', '--backup'])

    # Clone the required HARL repository
    subprocess.run(['git', 'clone', 'https://github.com/PKU-MARL/HARL.git'])
    # subprocess.call('cd HARL', shell=True)
    os.chdir('HARL')
    subprocess.run(['pip', 'install', '-e', '.'])
    # subprocess.call('cd ..', shell=True)
    os.chdir('..')

    # Move the modified files to the HARL repository
    subprocess.run(['python', 'utilities/move_modified_files.py'])


if __name__ == '__main__':
    main()
