"""Shared trial object for deployment."""

class DeploymentTrial:
    """Picklable trial object for paper trader."""
    def __init__(self, trial_data):
        self.number = trial_data['number']
        self.params = trial_data['params']
        self.user_attrs = trial_data['user_attrs']
        self.value = trial_data['sharpe']
