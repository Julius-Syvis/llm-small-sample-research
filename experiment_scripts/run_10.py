from models.core import get_supported_model_factories
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), get_supported_tasks(),
                                       TrainConfig(
                                           custom_train_sample_count=10,
                                           custom_eval_step=10,
                                           custom_max_step=200,
                                           experiment_name=f"100_samples"))
    sequencer.train()
