# anymal_rl

## Teacher policy training

```bash
python3 scripts/train.py --task=anymal_c_teacher --experiment=experiment_name
```

So far, the best policy is `ar5`

## Student policy training

```bash
python3 scripts/distill_with_noise_model.py --task=anymal_c_teacher --experiment=experiment_name
```

To later view the distilled policy, simply run the following command

```bash
python3 scripts/test_student.py
```