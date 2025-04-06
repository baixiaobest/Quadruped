import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

class Logger:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(list))  # key -> episode -> list of step values
        self.episodes = set()  # Track all unique episodes across keys

    def log(self, key, value, episode, step=None):
        """
        Log a value for a key, episode, and step.
        - If `value` is a list, it is treated as an entire episode (step is ignored).
        - If `value` is a scalar, `step` must be provided.
        """
        if isinstance(value, list):
            # Log an entire episode (list of step values)
            self.data[key][episode] = value
            self.episodes.add(episode)
        else:
            # Log a single step value
            if step is None:
                raise ValueError("For single values, `step` must be specified.")
            # Ensure the episode list is long enough to include this step
            while len(self.data[key][episode]) <= step:
                self.data[key][episode].append(None)  # Fill missing steps with None
            self.data[key][episode][step] = value
            self.episodes.add(episode)

    def get_keys(self):
        """Get all logged keys."""
        return list(self.data.keys())
    
    def get_data(self, key):
        """Get all data for a specific key."""
        if key not in self.data:
            raise ValueError(f"Key {key} not found in logger.")
        return self.data[key]

    def get_episode_values(self, key, episode):
        """Get values for a specific key and episode."""
        return self.data[key][episode]

    def get_all_episodes(self, key):
        """Get all episodes for a key, sorted by episode number."""
        episodes = sorted(self.data[key].keys())
        return [self.data[key][ep] for ep in episodes]
    def plot(self, key, x_axis='episode', aggregation='mean', episode=None, skip_none=True):
        """
        Plot logged data for a key.
        - `x_axis`: 'episode' (X = episode number) or 'step' (X = step number).
        - `aggregation`: 'mean', 'sum', 'max', 'min', or 'mean_std' (mean ± std, episode only).
        - `episode`: Required if x_axis='step' to specify which episode to plot.
        - `skip_none`: Whether to ignore None values.
        """
        if key not in self.data:
            raise ValueError(f"Key {key} not found in logger.")

        plt.figure()

        if x_axis == 'episode':
            # Aggregate across episodes
            episodes = sorted(self.data[key].keys())
            x, y_vals, y_stds = [], [], []

            for ep in episodes:
                values = self.data[key][ep]
                if skip_none:
                    values = [v for v in values if v is not None]
                
                if not values:  # Skip episodes with no valid data
                    continue
                    
                if aggregation == 'mean':
                    y_vals.append(np.mean(values))
                    x.append(ep)
                elif aggregation == 'sum':
                    y_vals.append(np.sum(values))
                    x.append(ep)
                elif aggregation == 'max':
                    y_vals.append(np.max(values))
                    x.append(ep)
                elif aggregation == 'min':
                    y_vals.append(np.min(values))
                    x.append(ep)
                elif aggregation == 'mean_std':
                    y_vals.append(np.mean(values))
                    y_stds.append(np.std(values))
                    x.append(ep)
                else:
                    raise ValueError("Invalid aggregation. Use 'mean', 'sum', 'max', 'min', or 'mean_std'.")

            plt.xlabel("Episode")
            if aggregation == 'mean_std':
                # Plot mean ± std with error bars
                plt.errorbar(x, y_vals, yerr=y_stds, fmt='-o', capsize=5, 
                            alpha=0.7, label='Mean ± 1 Std')
                plt.legend()
            else:
                plt.plot(x, y_vals, 'b-', alpha=0.6, label=f'{aggregation.capitalize()}')

        elif x_axis == 'step':
            # Plot steps for a specific episode
            if episode is None:
                raise ValueError("For x_axis='step', specify `episode`.")
            if episode not in self.data[key]:
                raise ValueError(f"Episode {episode} not found for key {key}.")

            values = self.data[key][episode]
            if skip_none:
                values = [v for v in values if v is not None]
            x = list(range(len(values)))
            y = values
            plt.plot(x, y, 'b-', alpha=0.6)
            plt.xlabel(f"Step in Episode {episode}")

        else:
            raise ValueError("Invalid x_axis. Use 'episode' or 'step'.")

        plt.ylabel("Value")
        plt.title(f"Logger Data for Key: {key} ({aggregation})")
        plt.grid(True)
        plt.legend()

    def save_to_file(self, filename):
        """
        Save logger data to a file using pickle.
        Handles defaultdict conversion for proper serialization.
        """
        # Convert nested defaultdicts to regular dicts for pickling
        data_dict = {key: dict(episodes) for key, episodes in self.data.items()}
        with open(filename, 'wb') as f:
            pickle.dump({
                'data': data_dict,
                'episodes': list(self.episodes)
            }, f)

    def load_from_file(self, filename):
        """
        Load logger data from a pickle file.
        Restores defaultdict structure after loading.
        """
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Clear existing data
        self.data.clear()
        self.episodes.clear()
        
        # Restore data structure
        data_dict = saved_data['data']
        for key, episodes in data_dict.items():
            self.data[key] = defaultdict(list, episodes)
        
        # Restore episodes
        self.episodes.update(saved_data['episodes'])

# Example Usage
if __name__ == "__main__":
    logger = Logger()

    # Log single values for episode 0
    logger.log('reward', 1.0, episode=0, step=0)
    logger.log('reward', 2.0, episode=0, step=1)

    # Log an entire episode (episode 1)
    logger.log('reward', [3.0, 4.0, 5.0], episode=1)

    # Log another key
    logger.log('loss', 0.5, episode=0, step=0)
    logger.log('loss', 0.3, episode=1, step=0)

    # Plot rewards aggregated by episode (mean)
    logger.plot('reward', x_axis='episode', aggregation='mean_std')

    # Plot raw rewards by step
    logger.plot('reward', x_axis='step', episode=0)

    logger.plot('loss', x_axis='episode', aggregation='sum')

    plt.show()