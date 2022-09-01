import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from melpy.run.runner import Runner as BaseRunner

from ..model.vgg import VGG16

def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot([labels, logits.shape[-1]])
    return -jnp.sum(jax.nn.log_softmax(logits)*one_hot, axis=-1)

class Runner(BaseRunner):
    def _fix_random(self):
        self.rng = jax.random.PRNGKey(self.cfg.random_seed)

    def _build_datasets(self):
        self.cfg.data.num_classes = 100
        ds = tfds.load(
            'cifar100', split='train', as_supervised=True, shuffle_files=True,
        )
        ds = ds.shuffle(1000).batch(32).prefetch(10).take(5)
        self.train_loader = ds

    def _build_models(self):
        self.model = VGG16(self.cfg.data.num_classes)

    def _build_losses(self):
        def loss_fn(images, labels):
            logits = self.model(images)
            return jnp.mean(softmax_cross_entropy(logits, labels))
        self.loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
        b = self.cfg.train.batch_size
        h = self.cfg.data.train_size
        w = self.cfg.data.train_size
        c = self.cfg.data.channels
        dummy_image = jnp.zeros((b, h, w, c))
        dummy_labels = jnp.zeros((b, self.cfg.data.num_classes))
        self.param = self.loss_fn.init(self.rng, dummy_image, dummy_labels)

    def _build_optimizers(self):
        def update_rule(param, update):
            return param - 0.01 * update
        self.optimizer = update_rule

    def _build_schedulers(self):
        ...

    def load(self):
        ...

    def train(self):
        for images, labels in self.train_loader:
            grads = jax.grad(self.loss_fn.apply)(self.params, images, labels)
            self.params = jax.tree_util.tree_map(
                self.optimizer, self.params, grads
            )