from models.core import get_xlm, get_roberta_base, get_big_bird, get_electra_base, get_supported_model_factories
from tasks.core import get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), [get_squad_v2()],
                                       [get_squad_v2()],
                                       TrainConfig(
                                           do_few_sample=False,
                                           experiment_name=f"full_samples_squad"))
    sequencer.train()
