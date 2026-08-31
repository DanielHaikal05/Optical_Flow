# NFlowNet DGX Commands

Commands to train the two proposed models for Normal Flow prediction: UNet and Small Raft

Run from `Optical_Flow` directory.

```bash
# Start training on DGX node 1
NFlowNet/scripts/dgx_train.sh

# Start RAFT training on DGX node 1
NFlowNet/scripts/dgx_train.sh --raft

# Live-follow the training log
NFlowNet/scripts/dgx_train.sh tail

# Show recent log lines once
NFlowNet/scripts/dgx_train.sh tail-once

# Check whether training is running
NFlowNet/scripts/dgx_train.sh status

# Stop training
NFlowNet/scripts/dgx_train.sh stop

# Sync only NFlowNet code
NFlowNet/scripts/dgx_train.sh sync-code

# Sync NFlowNet code and export train_raft.py as remote train.py
NFlowNet/scripts/dgx_train.sh sync-code --raft

# Sync TartanAir dataset
NFlowNet/scripts/dgx_train.sh sync-data --node 1

# Pull metric JSON files from DGX into NFlowNet/
NFlowNet/scripts/dgx_pull_metrics.sh
```

Defaults:
- Remote code: `~/Optical_Flow/NFlowNet`
- Remote data / `ROOT`: `~/Optical_Flow/Datasets/TartanAir`
- Remote log: `~/Optical_Flow/NFlowNet/logs/dgx/train.log`

Override dataset root:

```bash
NFlowNet/scripts/dgx_train.sh --root /path/to/TartanAir
```
