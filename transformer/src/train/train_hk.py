import functools
import haiku as hk
import optax
import jax
from logging import Logger
import time

from loss import lm_loss_fn
from forwardpass import build_forward_fn
from gradientupdate import GradientUpdater
from ..dataloader import JaxDataloader
from ..utils import TrainConfig


logging = Logger.getLogger('__main__')

MAX_STEPS: int = 1000
CONFIG: TrainConfig = TrainConfig(config_file="../train_config.json")

def train(num_heads: int = 4,
          d_model: int = 3,
          num_layers: int = 64,
          dropout_rate: float = 0.002,
          grad_clip_value: float = 0.001,
          learning_rate: float = 0.004) -> None:
    jdl: JaxDataloader = JaxDataloader(mongo_uri=CONFIG.mongo_uri,
                                       database_name=CONFIG.database_name,
                                       collection_names=CONFIG.collection_names,
                                       max_length=CONFIG.max_length,
                                       batch_size=CONFIG.batch_size,)

    train_dataset, vocab_size = jdl.run(), jdl.vocab_size()
    
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