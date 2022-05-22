from models.core import get_character_encoder_model_factories
from tasks.core import get_conll_2003, get_swag, get_ag_news, \
    get_squad_v2
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    for task in [get_squad_v2()]:
        sequencer = MultipleTrainSequencer(
            get_character_encoder_model_factories(),
            [task],
            TrainConfig(
                do_test_loop=True,
                experiment_name=f"canine"))
        sequencer.train()
