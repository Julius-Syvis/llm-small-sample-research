from models.core import get_xlm_roberta, get_multilingual_bert, get_multilingual_bert_cased, get_xlm_roberta_large
from tasks.core import get_wikiann_en, get_wikiann_lt
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        [get_xlm_roberta(), get_multilingual_bert(), get_multilingual_bert_cased()],
        [get_wikiann_en(), get_wikiann_lt()],
        TrainConfig(
            do_test_loop=True,
            experiment_name=f"multilinguals"))
    sequencer.train()
