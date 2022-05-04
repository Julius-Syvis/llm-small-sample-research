from models.core import get_bert_base, get_bert_base_uncased, get_roberta_base, get_electra_base, get_big_bird, get_xlm
from tasks.core import get_swag
from training.core import TrainConfig, TrainSequencer

if __name__ == "__main__":
    task = get_swag()
    train_config = TrainConfig(experiment_name="test_swag", do_test_overfit=True)

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
