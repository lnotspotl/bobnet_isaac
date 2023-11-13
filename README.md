# anymal_rl

## Teacher policy training

```bash
python3 scripts/train.py --task=anymal_c_teacher --experiment=experiment_name
```

So far, the best policy is `ar5`

## Student policy training

```bash
python3 scripts/distill_with_noise_model --task=anymal_c_teacher --experiment=experiment_name
```