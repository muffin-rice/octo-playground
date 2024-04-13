PPO with JaxMARL implementation with strong types. Checkpoints are stored in ckpt/*.ckpt

train.py is the training script which will generate checkpoints and use Torch Tensorboard writer to log loss metrics.

visualiser.py will generate a gif based on the vended checkpoint in visualizer_config 

We ditch Marimo because Jax on M1 is unstable and I like PyCharm.