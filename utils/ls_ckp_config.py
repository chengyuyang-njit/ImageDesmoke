# Show the configuration of the checkpoint
import argparse
import torch
import json

args = argparse.ArgumentParser(description="show configuration of checkpoint")
args.add_argument('-p', '--PATH', default = None, type = str, required = True,
                      help = 'path to the checkpoint file')
args = args.parse_args()

ckp = torch.load(args.PATH)

print(json.dumps(ckp["config"], indent = 4))