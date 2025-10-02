import json
import matplotlib.pyplot as plt

EXPERIMENTS = {
    'sync': {
        'path': 'sync_results.json',
        'label': 'Synchronous (FedAvg)',
        'color': 'red', 'marker': 'o', 'linestyle': '-'
    },
    'async_naive': {
        'path': 'basic_async_results.json',
        'label': 'Asynchronous (Naive)',
        'color': '#f39c12', 'marker': 'x', 'linestyle': '--' 
    },
    'async_linear': {
        'path': 'async_linear_staleness_decay_results.json',
        'label': 'Async (Linear Decay)',
        'color': '#8e44ad', 'marker': 'P', 'linestyle': ':' 
    },
    'async_poly': {
        'path': 'async_polynomial_staleness_decay_results.json',
        'label': 'Async (Polynomial Decay)',
        'color': '#2980b9', 'marker': 's', 'linestyle': '-.' 
    },
    'async_exp': {
        'path': 'async_exponential_staleness_decay_results.json',
        'label': 'Async (Exponential Decay)',
        'color': '#27ae60', 'marker': 'D', 'linestyle': '-' 
    },
    'async_inv_time': {
        'path': 'async_inverse_time_staleness_decay_results.json',
        'label': 'Async (Inverse-Time Decay)',
        'color': '#c0392b', 'marker': 'v', 'linestyle': '--' 
    },
    'async_adaptive': {
        'path': 'async_adaptive_staleness_results.json',
        'label': 'Async (Adaptive Threshold - Our Contribution)',
        'color': '#000000', 'marker': '*', 'linestyle': '-', 'linewidth': 3 
    }
}

def load_all_data(experiments_config):
    all_data = {}
    for key, config in experiments_config.items():
        try:
            with open(config['path'], 'r') as f:
                data = json.load(f)
                time_key = 'wall_clock_time'
                acc_key = 'accuracy'
                
                all_data[key] = {
                    'time': [0] + [item[time_key] for item in data],
                    'acc': [0] + [item[acc_key] * 100 for item in data]
                }
        except FileNotFoundError:
            print(f"Warning: Could not find result file '{config['path']}' for '{key}'. Skipping.")
    return all_data

def create_main_comparison_plot(data):
    keys_to_plot = ['sync', 'async_naive', 'async_exp']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in keys_to_plot:
        if key in data:
            config = EXPERIMENTS[key]
            ax.plot(data[key]['time'], data[key]['acc'], 
                    label=config['label'], color=config['color'], 
                    marker=config['marker'], linestyle=config['linestyle'], 
                    linewidth=2.5, markersize=8)

    ax.axhline(y=60.0, color='grey', linestyle='--', linewidth=1.5, label='60% Target Accuracy')
    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=14)
    ax.set_ylabel('Global Model Accuracy (%)', fontsize=14)
    ax.set_title('Convergence Comparison in a Light Straggler Environment', fontsize=18, pad=20)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(bottom=0) 
    ax.set_xlim(left=0)
    fig.tight_layout()
    
    output_filename = "figure_1_main_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Figure 1 saved as '{output_filename}'")

def create_ablation_study_plot(data):
    keys_to_plot = ['async_linear', 'async_poly', 'async_exp', 'async_inv_time', 'async_adaptive']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in keys_to_plot:
        if key in data:
            config = EXPERIMENTS[key]
            ax.plot(data[key]['time'], data[key]['acc'], 
                    label=config['label'], color=config.get('color'), 
                    marker=config.get('marker'), linestyle=config.get('linestyle'), 
                    linewidth=config.get('linewidth', 2), markersize=8)

    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=14)
    ax.set_ylabel('Global Model Accuracy (%)', fontsize=14)
    ax.set_title('Ablation Study of Asynchronous Staleness Mitigation Techniques', fontsize=18, pad=20)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(bottom=0) 
    ax.set_xlim(left=0)
    fig.tight_layout()
    
    output_filename = "figure_2_ablation_study.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Figure 2 saved as '{output_filename}'")

if __name__ == '__main__':
    all_results = load_all_data(EXPERIMENTS)
    
    if not all_results:
        print("No data was loaded. Aborting plot generation.")
    else:
        create_main_comparison_plot(all_results)
        create_ablation_study_plot(all_results)