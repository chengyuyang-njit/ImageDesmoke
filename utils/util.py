from datetime import datetime
import torch
import os

def _save_checkpoint(model, optimizer, total_loss, config, checkpoint_path, name=""):
    state = {
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'total_loss' : total_loss,
        'config' : config
    }
    now = str(datetime.now()).replace(" ","_")
    now = now.replace(":","_")
    filename = os.path.join(checkpoint_path , str(now)+str(name)+".pth")
    torch.save(state, filename)
    print("Saving checkpoint at time " + str(now))
