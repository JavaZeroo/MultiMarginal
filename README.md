# MultiMarginal

## Prepare

Link data 

```bash
ln -s /home/ljb/WassersteinSBP/data data
```

## Train

```bash
python train.py exp_name=gaussian2mnist dataset=mnist
```

Loss should be around 0.3 for `gaussian2mnist`

## Inference

```bash
python infer.py exp_name=gaussian2mnist checkpoint_path=/path/to/dir/of/checkpoint
```

example:
```bash
python infer.py exp_name=gaussian2mnist checkpoint_path=/home/ljb/WassersteinSBP/experiments/gaussian2mnist_new
```
