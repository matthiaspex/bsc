import argparse
parser = argparse.ArgumentParser()
parser.add_argument('T', type = int, help = 'temperature')

args = parser.parse_args()
T = args.T

print(f"Temperature is {T}")


print("Hello World")