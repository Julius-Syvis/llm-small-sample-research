from models.core import get_electra_base
from tasks.core import get_wikiann_en
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        [get_electra_base()],
        [get_wikiann_en()],
        TrainConfig(
            do_train=False,
            custom_train_sample_count=0,
            batch_size_multiplier=2,
            experiment_name=f"electra_wikiann_none"))
    sequencer.train()

    sequencer = MultipleTrainSequencer(
        [get_electra_base()],
        [get_wikiann_en()],
        TrainConfig(
            custom_train_sample_count=10,
            custom_eval_step=10,
            custom_max_step=200,
            early_stopping_patience=3,
            batch_size_multiplier=2,
            experiment_name=f"electra_wikiann_10"))
    sequencer.train()

    for sample_size in [10000]:
        sequencer = MultipleTrainSequencer(
            [get_electra_base()],
            [get_wikiann_en()],
            TrainConfig(
                custom_train_sample_count=sample_size,
                early_stopping_patience=5,
                validation_set_size_limit=100,
                test_set_size_limit=100,
                batch_size_multiplier=2,
                experiment_name=f"electra_wikiann_{sample_size}"))
        sequencer.train()
