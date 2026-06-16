# TML 2026 — Assignment 3: Adversarial Robustness
## Team: team_XXVI | atml_team024 | Team #27

## How to Recreate the Best Leaderboard Result (Score: 0.588054)

### Step 1 — Connect to cluster (requires university VPN)
```bash
ssh atml_team024@conduit2.hpc.uni-saarland.de
```

### Step 2 — Setup and download dataset
```bash
mkdir ~/tml26_task3 && cd ~/tml26_task3
mkdir runlogs
wget --header="Authorization: Bearer YOUR_HF_TOKEN" "https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz"
```

### Step 3 — Upload training script
```bash
scp train_robust.py atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
scp run.sh atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
scp task3.sub atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
```

### Step 4 — Submit training job (PGD adversarial training, ~5 hours)
```bash
cd ~/tml26_task3
chmod +x run.sh
condor_submit task3.sub
```

### Step 5 — Monitor training
```bash
watch condor_q
tail -f ~/tml26_task3/runlogs/task3.*.out
```

### Step 6 — Submit best model to leaderboard
Update `submission.py` with your API key, then:
```bash
scp submission.py atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
scp submit_model.sh atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
scp submit_job.sub atml_team024@conduit2.hpc.uni-saarland.de:~/tml26_task3/
condor_submit submit_job.sub
```

## Method Summary
- **Architecture**: Standard ResNet18 (final FC layer replaced for 9 classes), no architectural modifications to remain server-compatible
- **Adversarial training**: PGD-10 with ε=8/255, α=2/255, randomly initialized within the ε-ball
- **Evaluation attack**: PGD-20 for a reliable robustness estimate during validation
- **Data augmentation**: random horizontal flip + random crop (reflect-pad 4, crop 32×32)
- **Optimizer**: SGD with Nesterov momentum (0.9), weight decay 5e-4, initial LR 0.05
- **LR schedule**: MultiStepLR, decay by 10× at epochs 75, 110, 135, over 150 total epochs
- **Validation split**: 45,000 train / 5,000 held-out validation (fixed seed)

## Best Result
- **Score: 0.588054** (0.5·clean_acc + 0.5·robust_acc)
- **Validation**: clean=0.694, robust=0.478, score=0.586 (epoch 150)
- **Team: team_XXVI | Team #27**
