"""Separate actor input capability from scorer truth/QC capability."""
from .clock import immutable_array
from .timeindex import checked_index


class ActorStore:
    def __init__(self, read_current):
        self.__read_current = read_current

    def read_input(self, index, t_index):
        checked_index(index); checked_index(t_index)
        if index != t_index:
            raise PermissionError('actor can only open its current input')
        value = self.__read_current(index)
        return None if value is None else immutable_array(value)


class ScorerStore:
    def __init__(self, read_truth):
        self.__read_truth = read_truth

    def truth(self, index):
        checked_index(index)
        value = self.__read_truth(index)
        return None if value is None else immutable_array(value)


def actor_config(full):
    # Intentionally whitelist, rather than removing only today's known secrets.
    allowed = ('variables', 'lead_steps', 'model_version', 'fit_version', 'arms', 'tau_days')
    import copy
    return {key: copy.deepcopy(full[key]) for key in allowed if key in full}


def input_eligible(index, bad_indices):
    return checked_index(index) not in bad_indices


def scoring_mask(index, steps, bad_indices):
    return tuple(input_eligible(index, bad_indices) and index+k not in bad_indices
                 and index+k < 1464 for k in steps)


def feedback_eligible(index, bad_indices):
    return input_eligible(index, bad_indices) and all(index+k not in bad_indices and index+k < 1464 for k in (1,2,3,4))
