from models.core import get_bert_base
from tasks.core import get_conll_2003
from training.config import TrainConfig
from training.core import MultipleTrainSequencer

if __name__ == "__main__":
    for task in [get_conll_2003()]:
        sequencer = MultipleTrainSequencer(
            [get_bert_base()],
            [task],
            TrainConfig(
                do_test_loop=True,
                experiment_name=f"canine"))
        sequencer.train()
