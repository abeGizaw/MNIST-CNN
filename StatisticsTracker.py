import numpy as np
import os
import json

class StatisticsTracker:
    def __init__(self, model_name, stats_file):
        self.model_name = model_name
        self.times = {'step': [], 'epoch': [], 'training': [], 'validation': [], 'combined': []}
        self.epoch_accuracies = {}  # This will hold the accuracies for each epoch
        self.validation_accuracies = []  # This will hold the validation accuracies
        self.timesRan = 0
        self.stats_file = os.path.join('data', stats_file)
        os.makedirs('data', exist_ok=True)
        self.load_statistics()

    def load_statistics(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as file:
                data = json.load(file)
                self.timesRan = data.get('times ran', self.timesRan)
                self.times = data.get('times', self.times)
                self.epoch_accuracies = data.get('epoch_accuracies', self.epoch_accuracies)
                self.validation_accuracies = data.get('validation_accuracies', self.validation_accuracies)


    def save_statistics_to_json(self):
        data = {
            'times ran': self.timesRan,
            'times': self.times,
            'epoch_accuracies': {},
            'validation_accuracies': self.validation_accuracies
        }
        print(f'data we will add is{data}')

        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as file:
                existing_data = json.load(file)
            # Load existing epoch accuracies
            existing_epoch_accuracies = existing_data.get('epoch_accuracies', {})
        else:
            existing_epoch_accuracies = {}

        # Update the epoch accuracies with new data
        for epoch, accuracies in self.epoch_accuracies.items():
            if epoch in existing_epoch_accuracies:
                # Append the new accuracies to the existing list for this epoch
                existing_epoch_accuracies[epoch].extend([accuracies[-1]])
            else:
                # Create a new list for this epoch with the new accuracies
                existing_epoch_accuracies[epoch] = [accuracies[-1]]

        data['epoch_accuracies'] = existing_epoch_accuracies


        with open(self.stats_file, 'w') as file:
            json.dump(data, file, indent=4)

    def add_time(self, phase,elapsed_time):
        self.times[phase].append(elapsed_time)

    def record_epoch_accuracy(self, epoch, accuracy):
        if str(epoch) not in self.epoch_accuracies:
            self.epoch_accuracies[str(epoch)] = []
        self.epoch_accuracies[str(epoch)].append(accuracy)

    def record_validation_accuracy(self, accuracy):
        self.validation_accuracies.append(accuracy)

    def calculate_averages(self):
        averages = {phase: np.mean(times) for phase, times in self.times.items()}
        epoch_accuracy_averages = {epoch: np.mean(acc) for epoch, acc in self.epoch_accuracies.items()}
        averages['epoch_accuracy'] = epoch_accuracy_averages
        averages['validation_accuracy'] = np.mean(self.validation_accuracies)
        return averages

    def save_statistics(self):
        self.timesRan += 1
        self.save_statistics_to_json()
        averages = self.calculate_averages()
        with open(os.path.join('data', f'{self.model_name}_statistics.txt'), 'w') as file:
            file.write(f'Model: {self.model_name}\n')
            file.write(f'Num of times ran: {self.timesRan}\n')
            for phase, times in self.times.items():
                file.write(f'Avg {phase.capitalize()} Time: {np.mean(times):.2f}\n')
            for epoch, acc in averages['epoch_accuracy'].items():
                file.write(f'Avg Epoch {epoch} Accuracy: {acc:.4f}\n')
            file.write(f'Avg Validation Accuracy: {averages["validation_accuracy"]:.4f}\n')
