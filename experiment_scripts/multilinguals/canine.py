from models.core import get_canine_c, get_canine_s
from tasks.core import get_squad_v2, get_wikiann_lt, get_wikiann_en
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
            [get_canine_c(), get_canine_s()],
            [get_wikiann_en(), get_wikiann_lt()],
        TrainConfig(
            do_train=False,
            custom_train_sample_count=0,
            experiment_name=f"en_lt_comparison_none"))
    sequencer.train()

    sequencer = MultipleTrainSequencer(
            [get_canine_c(), get_canine_s()],
            [get_wikiann_en(), get_wikiann_lt()],
        TrainConfig(
            custom_train_sample_count=10,
            custom_eval_step=10,
            custom_max_step=200,
            early_stopping_patience=3,
            experiment_name=f"en_lt_comparison_10"))
    sequencer.train()
