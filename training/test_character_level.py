from models.core import get_canine_c, get_bert_base
from tasks.core import get_conll_2003
from training.core import TrainConfig, TrainSequencer

if __name__ == "__main__":
    task = get_conll_2003()
    model_factory = get_bert_base()
    train_config = TrainConfig(experiment_name="test_conll_chars", do_test_loop=True)

    sequencer = TrainSequencer(model_factory, task, train_config)
    sequencer.train()