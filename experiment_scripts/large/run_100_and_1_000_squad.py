from models.core import get_supported_model_factories, get_supported_large_model_factories
from tasks.core import get_supported_tasks, get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_large_model_factories(), [get_squad_v2()],
                                       TrainConfig(
                                           custom_train_sample_count=100,
                                           experiment_name=f"large"))
    sequencer.train()

    sequencer = MultipleTrainSequencer(get_supported_large_model_factories(), [get_squad_v2()],
                                       TrainConfig(
                                           custom_train_sample_count=1_000,
                                           experiment_name=f"large"))
    sequencer.train()
