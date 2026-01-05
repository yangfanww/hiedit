import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from data.base import make_loader
from model import make_model
import swanlab


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    swanlab.init(
        project="hiedit",
        experiment_name=f"{config.editor.cache_dir}_k{config.editor.n_layers}_{config.model.name}_{config.editor.name}_{config.dataset.name}_{config.num_seq}",
        config=OmegaConf.to_container(config, resolve=True)
    )
    data_module = importlib.import_module(f"data.{config.dataset.name}")
    data_class = getattr(data_module, f"{config.dataset.name.upper()}Dataset")

    train_loader, valid_loader = make_loader(config, data_class)

    model = make_model(config.model).to(config.model_device)

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    editor.run(train_loader, valid_loader)


if __name__ == "__main__":
    main()