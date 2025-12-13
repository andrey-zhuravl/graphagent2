import configparser

import tomli
import pathlib

def get_config_dict() -> dict:
    config_path = pathlib.Path(__file__).resolve().parents[2] / "config.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)
        #for key, value in config.items():
            #print(f"key = {key}: value = {value}")
        return config

if __name__ == "__main__":
    config = get_config_dict()