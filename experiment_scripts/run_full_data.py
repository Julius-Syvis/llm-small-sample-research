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
                do_few_sample=False,

                delete_after_save=True,
                batch_size_multiplier=4,
                experiment_name=f"full_samples_{task.hub_dataset_name}",
            )

            sequencer = TrainSequencer(model_factory, task, config)
            sequencer.train()
