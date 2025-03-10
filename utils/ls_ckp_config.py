# Show the configuration of the checkpoint
import argparse
import torch
import json

args = argparse.ArgumentParser(description="show configuration of checkpoint")
args.add_argument('-p', '--PATH', default = None, type = str, required = True,
                      help = 'path to the checkpoint file')
args = args.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckp = torch.load(args.PATH, map_location=device)

print(json.dumps(ckp["config"], indent = 4))