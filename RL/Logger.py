import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from enum import Enum

class KeyType(Enum):
    EPISODE = 1
    UPDATE = 2

class Logger:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(list))  # key -> identifier (episode/update_round) -> list of values
        self.key_type_map = {}  # Maps each key to KeyType
        self.episodes = set()  # Track all episodes for EPISODE keys
        self.update_rounds = set()  # Track all update rounds for UPDATE keys
        self.episode_offset = 0  # Offset for episode numbers
        self.update_round_offset = 0  # Offset for update rounds

    def log_episode(self, key, value, episode, step=None):
        """Log data for an episode-type key."""
        episode = episode + self.episode_offset
        self._log(key, value, KeyType.EPISODE, identifier=episode, step=step)

    def log_update(self, key, value, update_round, step=None):
        """Log data for an update-type key."""
        update_round = update_round + self.update_round_offset
        self._log(key, value, KeyType.UPDATE, identifier=update_round, step=step)\
        
    def get_max_episode(self):
        """Get the maximum episode number."""
        max_episode = 0
        for identifier in self.episodes:
            max_episode = max(max_episode, identifier)
        return max_episode
    
    def get_max_update_round(self):
        max_update_round = 0
        for identifier in self.update_rounds:
            max_update_round = max(max_update_round, identifier)
        return max_update_round
    
    def set_episode_offset(self, offset):
        """Set the offset for episode numbers."""
        self.episode_offset = offset
    
    def set_update_round_offset(self, offset):
        """Set the offset for update rounds."""
        self.update_round_offset = offset

    def _log(self, key, value, key_type, identifier, step=None):
        # Check key type consistency
        if key in self.key_type_map:
            if self.key_type_map[key] != key_type:
                raise ValueError(f"Key {key} is already registered as {self.key_type_map[key]}, cannot log as {key_type}.")
        else:
            self.key_type_map[key] = key_type

        # Track identifier
        if key_type == KeyType.EPISODE:
            self.episodes.add(identifier)
        else:
            self.update_rounds.add(identifier)

        # Handle logging
        if isinstance(value, list):
            self.data[key][identifier] = value
        else:
            if step is None:
                raise ValueError("For single values, `step` must be specified.")
            while len(self.data[key][identifier]) <= step:
                self.data[key][identifier].append(None)
            self.data[key][identifier][step] = value

    def get_keys(self):
        """Get all logged keys."""
        return list(self.data.keys())

    def get_key_type(self, key):
        """Get the KeyType of a key."""
        return self.key_type_map.get(key)

    def get_data(self, key):
        """Get all data for a specific key."""
        if key not in self.data:
            raise ValueError(f"Key {key} not found in logger.")
        return self.data[key]

    def get_episode_values(self, key, episode):
        """Get values for an episode-type key and episode."""
        if self.key_type_map.get(key) != KeyType.EPISODE:
            raise ValueError(f"Key {key} is not an episode-type key.")
        return self.data[key][episode]

    def get_all_episodes(self, key):
        """Get all episodes for an episode-type key."""
        if self.key_type_map.get(key) != KeyType.EPISODE:
            raise ValueError(f"Key {key} is not an episode-type key.")
        episodes = sorted(self.data[key].keys())
        return [self.data[key][ep] for ep in episodes]

    def plot_episode(self, key, aggregation='mean', episode=None, skip_none=True):
        """Plot data for an episode-type key."""
        if self.key_type_map.get(key) != KeyType.EPISODE:
            raise ValueError(f"Key {key} is not an episode-type key.")
        x_axis = 'step' if episode is not None else 'episode'
        xlabel = f"Step in Episode {episode}" if episode is not None else "Episode"
        self._plot(key, x_axis, aggregation, episode, skip_none, xlabel)

    def plot_update(self, key, aggregation='mean', update_round=None, skip_none=True):
        """Plot data for an update-type key."""
        if self.key_type_map.get(key) != KeyType.UPDATE:
            raise ValueError(f"Key {key} is not an update-type key.")
        x_axis = 'step' if update_round is not None else 'update_round'
        xlabel = f"Step in Update Round {update_round}" if update_round is not None  else "Update Round"
        self._plot(key, x_axis, aggregation, update_round, skip_none, xlabel)

    def _plot(self, key, x_axis, aggregation, identifier, skip_none, xlabel):
        if key not in self.data:
            raise ValueError(f"Key {key} not found in logger.")

        plt.figure()

        if x_axis in ['episode', 'update_round']:
            identifiers = sorted(self.data[key].keys())
            x, y_vals, y_stds = [], [], []

            for iden in identifiers:
                values = self.data[key][iden]
                if skip_none:
                    values = [v for v in values if v is not None]
                if not values:
                    continue

                if aggregation == 'mean':
                    y = np.mean(values)
                elif aggregation == 'sum':
                    y = np.sum(values)
                elif aggregation == 'max':
                    y = np.max(values)
                elif aggregation == 'min':
                    y = np.min(values)
                elif aggregation == 'mean_std':
                    y = np.mean(values)
                    y_std = np.std(values)
                    y_stds.append(y_std)
                else:
                    raise ValueError(f"Invalid aggregation: {aggregation}")

                x.append(iden)
                y_vals.append(y)

            plt.xlabel(xlabel)
            plt.ylabel("Value")
            plt.title(f"Logger Data for Key: {key} ({aggregation})")
            plt.grid(True)

            if aggregation == 'mean_std':
                plt.errorbar(x, y_vals, yerr=y_stds, fmt='-o', capsize=5, alpha=0.7, label='Mean ± 1 Std')
                plt.legend()
            else:
                plt.plot(x, y_vals, 'b-o', markersize=0.5, alpha=0.6, label=f'{aggregation.capitalize()}')
                plt.legend()

        elif x_axis == 'step':
            if identifier not in self.data[key]:
                raise ValueError(f"Identifier {identifier} not found for key {key}.")
            values = self.data[key][identifier]
            if skip_none:
                values = [v for v in values if v is not None]
            x_steps = list(range(len(values)))
            plt.plot(x_steps, values, 'b-o', markersize=0.5, alpha=0.6)
            plt.xlabel(xlabel)
            plt.ylabel("Value")
            plt.title(f"Logger Data for Key: {key} (Steps)")
            plt.grid(True)
        else:
            raise ValueError(f"Invalid x_axis: {x_axis}")

    def save_to_file(self, filename):
        """Save logger data to a file."""
        data_dict = {key: dict(identifier_dict) for key, identifier_dict in self.data.items()}
        with open(filename, 'wb') as f:
            pickle.dump({
                'data': data_dict,
                'key_type_map': self.key_type_map,
                'episodes': list(self.episodes),
                'update_rounds': list(self.update_rounds)
            }, f)

    def load_from_file(self, filename):
        """Load logger data from a file."""
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)

        self.data.clear()
        self.key_type_map.clear()
        self.episodes.clear()
        self.update_rounds.clear()

        for key, identifier_dict in saved_data['data'].items():
            self.data[key] = defaultdict(list, identifier_dict)

        self.key_type_map.update(saved_data['key_type_map'])
        self.episodes.update(saved_data.get('episodes', []))
        self.update_rounds.update(saved_data.get('update_rounds', []))

# Example Usage
if __name__ == "__main__":
    logger = Logger()

    # Log episode-type data
    logger.log_episode('reward', 1.0, episode=0, step=0)
    logger.log_episode('reward', 2.0, episode=0, step=1)
    logger.log_episode('reward', [3.0, 4.0, 5.0], episode=1)

    # Log update-type data
    logger.log_update('loss', 0.5, update_round=0, step=0)
    logger.log_update('loss', 0.3, update_round=1, step=0)
    logger.log_update('loss', [0.2, 0.1], update_round=2)

    # Plotting
    logger.plot_episode('reward', aggregation='mean_std')
    logger.plot_episode('reward', episode=0)

    logger.plot_update('loss', aggregation='sum')
    logger.plot_update('loss', update_round=1)

    plt.show()