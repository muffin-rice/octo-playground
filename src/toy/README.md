Equinox example as described here: https://docs.kidger.site/equinox/examples/mnist/

Has additional dependencies on Pytorch dataloaders and Optax optimizers. 

We also onboard to Marimo in `marimo_equinox_cifar10.py`: https://docs.marimo.io/getting_started/index.html. To run, just run `marimo edit src/toy/marimo_equinox_cifar10.py`. 

However, we require this code block for parity between `python marimo_equinox_cifar10.py` and running interactively:

```
import sys
import os
sys.path[0] = f'{os.getcwd()}/src/toy'
```

It also blocks * imports, which is another change from `equinox_cifar10.py`. We import every hyperparam separately.

Note: Somehow equinox_cifar10.py stopped working. Tried to upgrade back to jax 0.0.6 but it didn't work.