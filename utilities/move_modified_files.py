"""
This script moves the modified files from the resources/harl_modified directory to the HARL/harl directory.

"""

import os
import shutil


# Get the current working directory
cwd = os.getcwd()

# Files and directories
FILES = (
    "envs/__init__.py",
    "utils/configs_tools.py",
    "utils/envs_tools.py",
)

SOURCE = cwd + '/resources/harl_modified'
DESTINATION = cwd + '/HARL/harl'

errors = []


def main():
    # Check for errors
    if not os.path.exists(DESTINATION):  # whether the destination directory exists
        errors.append(f"{DESTINATION} does not exist.")

    if not all([os.path.exists(SOURCE + "/" + file) for file in FILES]):  # whether the files exist in the source directory
        errors.append(f"Files do not exist in {SOURCE}.")

    if not all([os.path.exists(DESTINATION + "/" + file) for file in FILES]):  # whether the files exist in the destination directory
        errors.append(f"Files do not exist in {DESTINATION}.")

    if errors:
        raise FileNotFoundError("ERRORS:\n- " + "\n- ".join(errors))

    # Copy the files from the source directory to the destination directory
    for file in FILES:
        source_file = SOURCE + "/" + file
        destination_file = DESTINATION + "/" + file

        os.rename(destination_file, destination_file + ".bak")
        shutil.copyfile(source_file, destination_file)
        
    print("Files moved successfully.")

    # Check if the files have been moved successfully
    for file in FILES:
        destination_file = DESTINATION + "/" + file
        if os.path.exists(destination_file):
            print(f"- The modified {destination_file} has been moved.")
        else:
            print(f"- The modified {destination_file} has not been added to the HARL repository.")


if __name__ == "__main__":
    main()
