from models.core import get_supported_model_factories, get_xlm
from tasks.core import get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_xlm()], [get_squad_v2()],
                                       TrainConfig(
                                           custom_train_sample_count=100,
                                           experiment_name=f"100_samples_squad"))
    sequencer.train()

    for train_size in [1000, 10_000, None]:
        sequencer = MultipleTrainSequencer(get_supported_model_factories(), [get_squad_v2()],
                                           TrainConfig(
                                               custom_train_sample_count=train_size,
                                               experiment_name=f"{train_size}_samples_squad"))
        sequencer.train()
