from models.core import get_bert_base, get_bert_base_uncased, get_roberta_base, get_electra_base, get_big_bird, get_xlm
from tasks.core import get_conll_2003, get_swag, get_ag_news
from training.core import TrainConfig, TrainSequencer

if __name__ == "__main__":
    for task in [get_conll_2003(), get_swag(), get_ag_news()]:
        config = TrainConfig(
            do_train=True,
            do_few_sample=False,

            batch_size_multiplier=2,
            experiment_name=f"full_samples_{task.hub_dataset_name}",
        )

        sequencer = TrainSequencer(get_bert_base(), task, config)
        sequencer.train()

        sequencer = TrainSequencer(get_bert_base_uncased(), task, config)
        sequencer.train()

        sequencer = TrainSequencer(get_roberta_base(), task, config)
        sequencer.train()

        sequencer = TrainSequencer(get_electra_base(), task, config)
        sequencer.train()

        sequencer = TrainSequencer(get_big_bird(), task, config)
        sequencer.train()

        sequencer = TrainSequencer(get_xlm(), task, config)
        sequencer.train()
