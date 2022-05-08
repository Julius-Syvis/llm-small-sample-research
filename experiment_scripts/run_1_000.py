from models.core import get_supported_model_factories
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), get_supported_tasks(),
                                       TrainConfig(
                                           do_train=True,
                                           custom_train_sample_count=1_000,

                                           batch_size_multiplier=4,
                                           experiment_name=f"1_000_samples"))
    sequencer.train()
