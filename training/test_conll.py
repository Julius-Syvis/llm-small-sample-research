from models.core import get_bert_base, get_roberta_base, get_bert_base_uncased, get_canine_s, get_canine_c, \
    get_electra_base, ModelFactory, get_big_bird, get_xlnet_base, get_transformer_xl, get_xlm
from tasks.core import get_conll_2003, NERTask
from training.core import TrainSequencer, TrainConfig
from utils.data_utils import prepare_test_dsd, shuffle_ds
from utils.gpu_utils import cleanup


if __name__ == "__main__":
    task = get_conll_2003()
    for do_train in [False]:
        train_config = TrainConfig(do_test_run=True, do_train=do_train)
        # TrainSequencer.train(get_bert_base(), task, train_config)
        # TrainSequencer.train(get_bert_base_uncased(), task, train_config)
        # TrainSequencer.train(get_roberta_base(), task, train_config)
        # TrainSequencer.train(get_canine_s(), task, train_config) # NEEDFIX
        # TrainSequencer.train(get_canine_c(), task, train_config) # NEEDFIX
        # TrainSequencer.train(get_electra_base(), task, train_config)
        # TrainSequencer.train(get_big_bird(), task, train_config)
        # TrainSequencer.train(get_xlnet_base(), task, train_config)
        # TrainSequencer.train(get_transformer_xl(), task, train_config) # NEEDFIX
        # TrainSequencer.train(get_xlm(), task, train_config)
