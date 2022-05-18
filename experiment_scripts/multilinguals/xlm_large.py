from models.core import get_xlm_roberta, get_xlm_roberta_large
from tasks.core import get_wikiann_en, get_wikiann_lt
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    # sequencer = MultipleTrainSequencer(
    #     [get_xlm_roberta_large()],
    #     [get_wikiann_en(), get_wikiann_lt()],
    #     TrainConfig(
    #         do_train=False,
    #         custom_train_sample_count=0,
    #         experiment_name=f"en_lt_comparison_none"))
    # sequencer.train()
    #
    # sequencer = MultipleTrainSequencer(
    #     [get_xlm_roberta_large()],
    #     [get_wikiann_en(), get_wikiann_lt()],
    #     TrainConfig(
    #         custom_train_sample_count=10,
    #         custom_eval_step=10,
    #         custom_max_step=200,
    #         early_stopping_patience=3,
    #         experiment_name=f"en_lt_comparison_10"))
    # sequencer.train()

    for sample_size in [10000]:
        sequencer = MultipleTrainSequencer(
            [get_xlm_roberta_large()],
            [get_wikiann_en(), get_wikiann_lt()],
            TrainConfig(
                custom_train_sample_count=sample_size,
                early_stopping_patience=5,
                validation_set_size_limit=100,
                test_set_size_limit=100,
                experiment_name=f"en_lt_comparison"))
        sequencer.train()
