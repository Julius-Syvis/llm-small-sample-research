from models.core import get_electra_base, get_big_bird, get_xlm
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_electra_base(), get_big_bird(), get_xlm()], get_supported_tasks(),
                                       TrainConfig(
                                           custom_train_sample_count=10_000,
                                           batch_size_multiplier=4,
                                           experiment_name=f"10_000_samples"))
    sequencer.train()
