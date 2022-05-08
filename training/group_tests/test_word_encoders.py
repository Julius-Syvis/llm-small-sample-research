from models.core import get_word_encoder_model_factories
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_word_encoder_model_factories(), get_supported_tasks(),
                                       TrainConfig(experiment_name="test_words", do_test_loop=True))
    sequencer.train()
