from models.core import get_canine_c, get_canine_s
from tasks.core import get_squad_v2, get_wikiann_lt, get_wikiann_en
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    for sample_size in [10000]:
        sequencer = MultipleTrainSequencer(
            [get_canine_c(), get_canine_s()],
            [get_wikiann_en(), get_wikiann_lt()],
            TrainConfig(
                custom_train_sample_count=sample_size,
                early_stopping_patience=5,
                validation_set_size_limit=100,
                test_set_size_limit=100,
                batch_size_multiplier=2,
                experiment_name=f"en_lt_comparison"))
        sequencer.train()
