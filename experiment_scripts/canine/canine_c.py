from models.core import get_canine_c
from tasks.core import get_squad_v2
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        [get_canine_c()],
        [get_squad_v2()],
        TrainConfig(
            do_train=False,
            custom_train_sample_count=0,
            batch_size_multiplier=2,
            experiment_name=f"canine_c_none"))
    sequencer.train()

    sequencer = MultipleTrainSequencer(
        [get_canine_c()],
        [get_squad_v2()],
        TrainConfig(
            custom_train_sample_count=10,
            custom_eval_step=10,
            custom_max_step=200,
            early_stopping_patience=3,
            batch_size_multiplier=2,
            experiment_name=f"canine_c_10"))
    sequencer.train()
