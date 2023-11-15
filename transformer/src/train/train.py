import jax
import jax.numpy as np
import optax
import multiprocessing

from ..model.transformer import Transformer
from ..dataloader.jax_dataloader import JaxDataloader
import logging


logger: logging.Logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, mongo_uri:str, database_name:str, collection_names:list, max_length:int=512):
        self.dataloader: JaxDataloader = JaxDataloader(mongo_uri=mongo_uri,
                                        database_name=database_name, 
                                        collection_names=collection_names)
        self.model: Transformer = Transformer(hidden_dim=4,
                                 num_heads=2,
                                 num_layers=4)
        self.max_length: int = max_length

    def create_optimizer(self, params):
        optimizer_def = optax.adam(learning_rate=1e-4)
        optimizer = optimizer_def.create(params)
        return optimizer

    @staticmethod
    @jax.jit
    def forward(params, model, batch):
        return model.apply(params, batch)

    @staticmethod
    @jax.jit
    def train_step(params, model, optimizer, batch):
        def loss_fn(params):
            logits = Trainer.forward(params, model, batch)
            # TODO: Implement loss function here
            loss = np.sum(logits)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss

    def train_model(self, dataset):
        rng = jax.random.PRNGKey(42)
        params = self.model.init(rng, np.ones((1, self.max_length, 1)))

        optimizer = self.create_optimizer(params)

        # loop
        num_epochs: int = 100000
        for epoch in range(num_epochs):
            for batch in dataset:
                rng, subkey = jax.random.split(rng)
                optimizer, loss = self.train_step(params, self.model, optimizer, batch)

            logger(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

        # save model
        jax.save_model("./checkpoints", params)

    def run(self) -> None:
        with multiprocessing.Pool() as pool:
            all_data: list[list[list]] = pool.map(self.dataloader.load_and_process_data, self.collection_names)

        flattened_batches: list[list] = [item for sublist in all_data for item in sublist]

        jax_dataset: list[list[list]] = self.dataloader.create_batches(flattened_batches)

        self.train_model(jax_dataset)
