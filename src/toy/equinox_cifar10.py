import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import optax  # https://github.com/deepmind/optax
import numpy as np
import torch  # https://pytorch.org
from torch.utils.data import DataLoader
import torchvision  # https://pytorch.org
from cifar10_hyperparams import *

key = jax.random.PRNGKey(SEED)

class CNN(eqx.Module):
    layers: []

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.layers = [
            eqx.nn.Conv2d(1, 3, 3, key=key1),
            eqx.nn.MaxPool2d(2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1875, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512,64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10,  key=key4),
            jax.nn.softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)

        return x

def loss(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Float[Array, "batch"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

def cross_entropy(y: Float[Array, "batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
    pred_y = jnp.log(jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1))
    return -jnp.mean(pred_y)

def numpy_collate(data):
    return torch.stack([pair[0] for pair in data], dim = 0).numpy(), np.array([pair[1] for pair in data])

@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, "batch"]
) -> Float[Array, ""]:
    return jnp.mean(y == jnp.argmax(jax.vmap(model)(x), axis=1))

def evaluate_dataset(model: CNN, test_loader: DataLoader) -> (Float[Array, ""], Float[Array, ""]):
    '''Returns Loss and Accuracy for model and DataLoader'''
    accuracy, losses = [], []
    for x, y in test_loader:
        losses.append(loss(model, x, y))
        accuracy.append(compute_accuracy(model, x, y))

    return jnp.mean(np.array(losses)), jnp.mean(np.array(accuracy))

def train(model: CNN,
          train_loader: DataLoader,
          test_loader: DataLoader,
          optim: optax.GradientTransformation,
          steps: int = STEPS,
          print_every: int = PRINT_EVERY) -> CNN:

    @eqx.filter_jit
    def make_step(
            model: CNN,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def infinite_trainloader():
        while True:
            yield from train_loader

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if ((step % print_every) == 0) or (step == (steps-1)):
            test_loss, test_accuracy = evaluate_dataset(model, test_loader)
            print(f'Step {step}: test loss {test_loss} and test accuracy {test_accuracy}')

    return model


if __name__ == "__main__":
    normalize_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=normalize_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=normalize_data,
    )

    train_loader = DataLoader(
        train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=numpy_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=numpy_collate
    )

    dummy_x, dummy_y = next(iter(train_loader))
    print(f'Shapes of dataloader:\n x: {dummy_x.shape}\n y: {dummy_y.shape}')

    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)
    print(f'Model tree: {model}')

    loss_value = loss(model, dummy_x, dummy_y)
    print(f'Loss shape: {loss_value}')

    output = jax.vmap(model)(dummy_x)
    print(f'Sample output: {output}')

    value, grads = eqx.filter_value_and_grad(loss)(model, dummy_x, dummy_y)

    optim = optax.adam(LEARNING_RATE)
    model = train(model, train_loader, test_loader, optim)