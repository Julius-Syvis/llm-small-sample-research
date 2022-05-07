from models.core import get_bert_base
from tasks.core import get_conll_2003
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_bert_base], [get_conll_2003],
                                       TrainConfig(experiment_name="test_stability", do_test_overfit=True,
                                                   num_runs=5, delete_after_save=True))
    sequencer.train()
