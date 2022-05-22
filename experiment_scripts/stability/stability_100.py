from models.core import get_electra_base, get_bert_base
from tasks.core import get_wikiann_en, get_ag_news
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        [get_bert_base()],
        [get_ag_news()],
        TrainConfig(
            custom_train_sample_count=100,
            early_stopping_patience=3,
            validation_set_size_limit=300,
            experiment_name=f"stability",
            num_runs=5,
            do_stability_test=True))
    sequencer.train()
