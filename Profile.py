class Profile():
    def __init__(self):
        self.number_episodes_til_solved=-1
        self.total_running_time=-1

        self.total_number_learn_calls=0
        self.total_number_sampling_calls=0
        self.total_number_evaluate_calls=0

        self.total_time_sampling=0
        self.total_time_learning=0
        self.total_time_evaluation=0
        self.total_time_training=0
        self.total_time_soft_update=0
        self.total_time_samples_to_environment_values=0
        self.total_time_updating_priorities=0
        self.total_time_introducing_isw=0
