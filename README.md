# anymal_rl

## Teacher policy training

```bash
python3 scripts/train.py --task=... --experiment=...
```

## Student policy training

```bash
python3 scripts/distill_with_noise_model.py --task=... --experiment=... --policy-path=...
```

# Credits
This repository is based on `legged_gym` from RSL at ETH: https://github.com/leggedrobotics/legged_gym
They are given full credit, I've merely made a couple of changes.