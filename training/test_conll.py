from models.core import get_bert_base, get_roberta_base, get_bert_base_uncased, get_canine_s, get_canine_c, \
    get_electra_base, get_big_bird, get_xlnet_base, get_transformer_xl, get_xlm
from tasks.core import get_conll_2003
from training.core import TrainSequencer, TrainConfig


if __name__ == "__main__":
    task = get_conll_2003()
    train_config = TrainConfig(experiment_name="test_conll", do_test_loop=True)

    sequencer = TrainSequencer(get_bert_base(), task, train_config)
    sequencer.train()

    sequencer = TrainSequencer(get_bert_base_uncased(), task, train_config)
    sequencer.train()

    sequencer = TrainSequencer(get_roberta_base(), task, train_config)
    sequencer.train()

    # sequencer = TrainSequencer(get_canine_s(), task, train_config) # NEEDFIX
    # sequencer.train()

    # sequencer = TrainSequencer(get_canine_c(), task, train_config) # NEEDFIX
    # sequencer.train()

    sequencer = TrainSequencer(get_electra_base(), task, train_config)
    sequencer.train()

    sequencer = TrainSequencer(get_big_bird(), task, train_config)
    sequencer.train()

    # sequencer = TrainSequencer(get_xlnet_base(), task, train_config) # NEEDFIX - gradient checkpointing
    # sequencer.train()

    # sequencer = TrainSequencer(get_transformer_xl(), task, train_config) # NEEDFIX
    # sequencer.train()

    sequencer = TrainSequencer(get_xlm(), task, train_config)
    sequencer.train()
