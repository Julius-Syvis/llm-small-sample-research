from models.core import get_supported_model_factories
from tasks.core import get_supported_tasks
from training.core import TrainConfig, TrainSequencer

if __name__ == "__main__":
    for t in get_supported_tasks():
        task = t()
        for mf in get_supported_model_factories():
            model_factory = mf()

            config = TrainConfig(
                do_train=True,
                do_few_sample=True,

                batch_size_multiplier=2,
                experiment_name=f"1000_samples_{task.hub_dataset_name}",
            )

            sequencer = TrainSequencer(model_factory, task, config)
            sequencer.train()
