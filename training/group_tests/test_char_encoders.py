from models.core import get_canine_c
from tasks.core import get_conll_2003
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_canine_c()], [get_conll_2003()],
                                       TrainConfig(experiment_name="test_chars", do_test_loop=True))
    sequencer.train()
