from models.core import get_supported_model_factories, get_supported_large_model_factories
from tasks.core import get_supported_tasks, get_squad_v2
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer(get_supported_model_factories() + get_supported_large_model_factories(),
                                       [get_squad_v2()],
                                       TrainConfig(
                                           do_train=False,
                                           custom_train_sample_count=0,
                                           experiment_name=f"no_samples"))
    sequencer.train()

    sequencer = MultipleTrainSequencer(get_supported_model_factories() + get_supported_large_model_factories(),
                                       [get_squad_v2()],
                                       TrainConfig(
                                           custom_train_sample_count=10,
                                           custom_eval_step=10,
                                           custom_max_step=200,
                                           early_stopping_patience=3,
                                           experiment_name=f"10_samples"))
    sequencer.train()
