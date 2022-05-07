from models.core import get_supported_model_factories
from tasks.core import get_swag
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), [get_swag],
                                       TrainConfig(experiment_name="test", do_test_loop=True))
    sequencer.train()
