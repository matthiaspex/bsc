from bsc_utils.miscellaneous import load_config_from_yaml, complete_sensor_selection

config = load_config_from_yaml("config\\general_bsc.yaml")

print(config["environment"]["sensor_selection"])
config = complete_sensor_selection(config)

print(config["environment"]["sensor_selection"])