import functools
import haiku as hk
import optax
import jax
from logging import Logger
import time

from loss import lm_loss_fn
from forwardpass import build_forward_fn
from gradientupdate import GradientUpdater


logging = Logger.getLogger('__main__')

MAX_STEPS = 1000

def train():
    train_dataset, vocab_size = load(batch_size,
                                     sequence_length)
    
    # Set up the model, loss, and updater
    forward_fn = build_forward_fn(vocab_size, d_model, num_heads,
                                  num_layers, dropout_rate)
    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99))

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    # Initialize parameters
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)