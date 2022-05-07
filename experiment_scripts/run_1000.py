from models.core import get_supported_model_factories
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), get_supported_tasks(),
                                       TrainConfig(
                                           do_train=True,
                                           do_few_sample=True,
                                           custom_train_sample_count=100,

                                           delete_after_save=True,
                                           batch_size_multiplier=4,
                                           experiment_name=f"1000_samples"))
    sequencer.train()
