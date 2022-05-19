from models.core import get_character_encoder_model_factories, get_bert_base
from tasks.core import get_conll_2003, get_swag, get_ag_news, \
    get_squad_v2
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        [get_bert_base()],
        [get_squad_v2()],
        TrainConfig(
            do_test_loop=True,
            experiment_name=f"squad"))
    sequencer.train()
