from models.core import get_supported_model_factories, get_xlm
from tasks.core import get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_xlm()], [get_squad_v2()],
                                       TrainConfig(
                                           custom_train_sample_count=10_000,
                                           experiment_name=f"10_000_samples_squad"))
    sequencer.train()
