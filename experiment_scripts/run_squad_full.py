from models.core import get_supported_model_factories, get_xlm
from tasks.core import get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories(), [get_squad_v2()],
                                       TrainConfig(
                                           do_few_sample=False,
                                           experiment_name=f"full_samples_squad"))
    sequencer.train()
