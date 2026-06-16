#!/bin/bash
pip install torch torchvision -q 2>/dev/null || true
/opt/conda/bin/python /home/atml_team024/tml26_task3/train_robust.py
