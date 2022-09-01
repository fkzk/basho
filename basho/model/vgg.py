import haiku as hk
import jax
import jax.numpy as jnp

class VGG16(hk.Module):
    def __init__(self, num_classes, name=None):
        super().__init__(name=name)
        self.layers = hk.Sequential([
            hk.Conv2D(64, 3), jax.nn.relu,
            hk.Conv2D(64, 3), jax.nn.relu,
            hk.MaxPool(2, 2, 'SAME'),
            hk.Conv2D(128, 3), jax.nn.relu,
            hk.Conv2D(128, 3), jax.nn.relu,
            hk.MaxPool(2, 2, 'SAME'),
            hk.Conv2D(256, 3), jax.nn.relu,
            hk.Conv2D(256, 3), jax.nn.relu,
            hk.Conv2D(256, 3), jax.nn.relu,
            hk.MaxPool(2, 2, 'SAME'),
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.MaxPool(2, 2, 'SAME'),
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.Conv2D(512, 3), jax.nn.relu,
            hk.MaxPool(14, 14, 'SAME'), hk.Flatten(),
            hk.Linear(4096), jax.nn.relu,
            hk.Linear(4096), jax.nn.relu,
            hk.Linear(10),
        ])

    def __call__(self, x):
        return self.layers(x)

def predict(x):
    model = VGG16(100)
    return model(x)

if __name__=='__main__':
    t_predict = hk.transform(predict)

    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    rng = jax.random.PRNGKey(42)
    params = t_predict.init(rng, x)
    y = t_predict.apply(params, rng, x)
    print(f'{jnp.shape(x)=}')
    print(f'{jnp.shape(y)=}')