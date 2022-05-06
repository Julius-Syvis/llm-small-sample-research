from models.core import get_bert_base
from tasks.core import get_conll_2003
from training.core import TrainConfig, TrainSequencer

if __name__ == "__main__":
    task = get_conll_2003()
    train_config = TrainConfig(experiment_name="test_stability", do_test_overfit=True, num_runs=5, delete_after_save=True)

    sequencer = TrainSequencer(get_bert_base(), task, train_config)
    sequencer.train()
