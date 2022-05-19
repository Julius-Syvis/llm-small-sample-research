from models.core import get_character_encoder_model_factories
from tasks.core import get_conll_2003, get_swag, get_ag_news, \
    get_squad_v2, get_supported_tasks
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(
        get_character_encoder_model_factories(),
        get_supported_tasks(),
        TrainConfig(
            do_test_overfit=True,
            experiment_name=f"canine"))
    sequencer.train()
