from models.core import get_supported_model_factories, get_xlnet_base
from tasks.core import get_supported_tasks
from training.core import TrainConfig, MultipleTrainSequencer

if __name__ == "__main__":
    sequencer = MultipleTrainSequencer([get_xlnet_base()],
                                       get_supported_tasks(),
                                       TrainConfig(
                                           do_train=False,
                                           custom_train_sample_count=0,
                                           experiment_name=f"no_samples_xlnet"))
    sequencer.train()

    sequencer = MultipleTrainSequencer([get_xlnet_base()],
                                       get_supported_tasks(),
                                       TrainConfig(
                                           custom_train_sample_count=10,
                                           custom_eval_step=10,
                                           custom_max_step=200,
                                           early_stopping_patience=3,
                                           experiment_name=f"10_samples_xlnet"))
    sequencer.train()

    for train_size in [100, 1_000, 10_000]:
        sequencer = MultipleTrainSequencer([get_xlnet_base()], get_supported_tasks(),
                                           TrainConfig(
                                               custom_train_sample_count=train_size,
                                               experiment_name=f"{train_size}_samples_xlnet"))
        sequencer.train()
