import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class FederatedLearningVisualizer:
    def __init__(self):
        self.metrics = {
            'global': {'loss': [], 'accuracy': [], 'auc': []},
            'clients': {}
        }

    def add_global_metrics(self, round_num, loss, accuracy, auc):
        self.metrics['global']['loss'].append((round_num, loss))
        self.metrics['global']['accuracy'].append((round_num, accuracy))
        self.metrics['global']['auc'].append((round_num, auc))

    def add_client_metrics(self, client_id, round_num, loss, accuracy, auc):
        if client_id not in self.metrics['clients']:
            self.metrics['clients'][client_id] = {'loss': [], 'accuracy': [], 'auc': []}
        self.metrics['clients'][client_id]['loss'].append((round_num, loss))
        self.metrics['clients'][client_id]['accuracy'].append((round_num, accuracy))
        self.metrics['clients'][client_id]['auc'].append((round_num, auc))

    def plot_global_metrics(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        rounds, losses = zip(*self.metrics['global']['loss'])
        ax1.plot(rounds, losses)
        ax1.set_title('Global Model Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')

        rounds, accuracies = zip(*self.metrics['global']['accuracy'])
        ax2.plot(rounds, accuracies)
        ax2.set_title('Global Model Accuracy')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')

        rounds, aucs = zip(*self.metrics['global']['auc'])
        ax3.plot(rounds, aucs)
        ax3.set_title('Global Model AUC')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('AUC')

        plt.tight_layout()
        plt.savefig('global_metrics.png')
        plt.close()

    def plot_client_metrics(self):
        metrics = ['loss', 'accuracy', 'auc']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
        
        for i, metric in enumerate(metrics):
            for client_id, client_metrics in self.metrics['clients'].items():
                rounds, values = zip(*client_metrics[metric])
                axes[i].plot(rounds, values, label=f'Client {client_id}')
            
            axes[i].set_title(f'Client {metric.capitalize()}')
            axes[i].set_xlabel('Round')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()

        plt.tight_layout()
        plt.savefig('client_metrics.png')
        plt.close()

    def plot_final_comparison(self):
        metrics = ['loss', 'accuracy', 'auc']
        client_ids = list(self.metrics['clients'].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            final_values = [self.metrics['clients'][client_id][metric][-1][1] for client_id in client_ids]
            global_final = self.metrics['global'][metric][-1][1]
            
            axes[i].bar(client_ids, final_values)
            axes[i].axhline(y=global_final, color='r', linestyle='--', label='Global Model')
            axes[i].set_title(f'Final {metric.capitalize()}')
            axes[i].set_xlabel('Client ID')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()

        plt.tight_layout()
        plt.savefig('final_comparison.png')
        plt.close()