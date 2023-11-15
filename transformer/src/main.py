from .utils.utils import TrainConfig
from .dataloader.jax_dataloader import JaxDataloader
from .model.transformer import Transformer
from .train.train import Trainer


def main(train_config:str="config.json") -> None:
    # load config from JSON
    config = TrainConfig(config_file=train_config)

    # create JAX dataloader
    dataloader: JaxDataloader = JaxDataloader(
        config.mongo_uri,
        config.database_name,
        config.collection_names,
        max_length=config.max_length,
        batch_size=config.batch_size
    )

    # Create Transformer model
    transformer_model: Transformer = Transformer(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate
    )

    # create and run training
    trainer = Trainer(dataloader, transformer_model)
    trainer.run()


if __name__ == "__main__":
    main()
