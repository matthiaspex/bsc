import yaml

with open("config\\test.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

print(cfg)