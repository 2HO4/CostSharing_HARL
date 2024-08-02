"""Train an algorithm."""
import os
import argparse
import json
import yaml
from harl.utils.configs_tools import update_args

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
            "custom"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo, or custom.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="default",
        choices=[
            "default",
            "custom"
        ],
        help="Environment type. Choose from: default or custom.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
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
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        if args['algo'] == 'custom':
            algo_cfg_path = os.path.join("resources", "configs", f"algorithm.yaml")
            with open(algo_cfg_path, "r", encoding="utf-8") as file:
                algo_args = yaml.load(file, Loader=yaml.FullLoader)
            args['algo'] = algo_args['algorithm']['name']
        else:
            algo_cfg_path = os.path.join( "HARL", "harl", "configs", "algos_cfgs", f"{args['algo']}.yaml")
            with open(algo_cfg_path, "r", encoding="utf-8") as file:
                algo_args = yaml.load(file, Loader=yaml.FullLoader)

        if args['env'] == 'custom':
            env_cfg_path = os.path.join("resources", "configs", f"environment.yaml")
            with open(env_cfg_path, "r", encoding="utf-8") as file:
                env_args = yaml.load(file, Loader=yaml.FullLoader)
        elif args['env'] == 'default':
            env_args = {arg: None for arg in ['details_firms', 'date_start', 'date_end', 'emissions_max', 'seed', 'price_is_constant', 'price_ceiling', 'price_floor', 'discount_demand', 'portion_cost', 'portion_reward', 'portion_punishment', 'reward_final', 'quota_production', 'cap_production', 'hide_others_moves', 'cost_function', 'algorithm', 'render_mode']}
        args['env'] = 'cost_sharing'
    
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    
    # import the runner
    from harl.runners import RUNNER_REGISTRY
    
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    
    # render the environment
    if algo_args["render"]["use_render"]:
        runner.run()
        runner.close()
        return
    
    # train the model
    continue_running = True
    
    while continue_running:
        runner.run()
        
        continue_running = input("Continue training? (y/n): ") == "y"
    
    runner.close()

    return


if __name__ == "__main__":
    main()
