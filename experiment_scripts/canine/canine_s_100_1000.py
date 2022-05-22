from models.core import get_canine_s
from tasks.core import get_squad_v2
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    for sample_size in [100, 1000]:
        sequencer = MultipleTrainSequencer(
            [get_canine_s()],
            [get_squad_v2()],
            TrainConfig(
                custom_train_sample_count=sample_size,
                early_stopping_patience=5,
                validation_set_size_limit=100,
                test_set_size_limit=100,
                batch_size_multiplier=2,
                experiment_name=f"canine_s_{sample_size}"))
        sequencer.train()
