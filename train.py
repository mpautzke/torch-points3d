#########################################
##                                     ##
##      NEXPLORE INNOVATION USA        ##
##                                     ##
## Azure Cloud Training Manager Script ##
##                                     ##
#########################################

_LOGO = """
      _   __                __
     / | / /__  _  ______  / /___  ________
    /  |/ / _ \\| |/_/ __ \\/ / __ \\/ ___/ _ \\
   / /|  /  __/>  </ /_/ / / /_/ / /  /  __/
  /_/ |_/\\___/_/|_/ .___/_/\\____/_/   \\___/
                 /_/
"""

print (_LOGO)

import argparse

print("Imports...")
print("  > hydra",flush=True)
import hydra
from omegaconf import OmegaConf
print("  > OmegaConf",flush=True)
from torch_points3d.trainer import Trainer
print("  > torch_points3d.trainer",flush=True)

@hydra.main(config_path="conf/config.yaml")
def main(cfg,args):

    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    print("Starting training preprocessing")
    trainer = Trainer(cfg)
    if not args.no_train:
        print("Starting training process NOW")
        trainer.train()
    else:
        print("Training is not being performed - --no-training option given.")

    # https://github.com/facebookresearch/hydra/issues/440
    hydra._internal.hydra.GlobalHydra.get_state().clear()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--no-training",action="store_true",dest="no_train",help="Do not perform training (only perform preprocessing)")

    args = parser.parse_args()
    main(args)
