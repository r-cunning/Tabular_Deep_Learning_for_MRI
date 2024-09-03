import matplotlib.pyplot as plt
import numpy as np

def plot_report(report_dict):
    # Extract keys, ground truths, predictions, errors, and baselines
    keys = list(report_dict.keys())
    ground_truths = np.array([report_dict[key]['ground_truth'] for key in keys]).flatten()
    predictions = np.array([report_dict[key]['prediction'] for key in keys]).flatten()
    model_errors = np.array([report_dict[key]['error'] for key in keys]).flatten()
    baselines = np.array([report_dict[key]['baseline'] for key in keys]).flatten()
    baseline_errors = np.array([report_dict[key]['baseline_error'] for key in keys]).flatten()
    # Define the positions of the bars
    bar_width = 0.3
    bar_positions = np.arange(len(keys))

    print(bar_positions.shape)

    print(f"bar_positions: {bar_positions.shape}")
    print(f"baselines: {np.shape(baselines)}")
    print(f"ground_truths: {np.shape(ground_truths)}")
    print(f"predictions: {np.shape(predictions)}")
    print(f"errors: {np.shape(model_errors)}")

    # Create figure with three subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 15))

    
    # First subplot: Baseline vs Ground Truth vs Prediction
    axs[0].bar(bar_positions - bar_width, baselines, width=bar_width, label='Baseline')
    axs[0].bar(bar_positions, ground_truths, width=bar_width, label='Ground Truth')
    axs[0].bar(bar_positions + bar_width, predictions, width=bar_width, label='Prediction')
    axs[0].set_title('Ground Truth vs Prediction')
    axs[0].set_ylabel('Values')
    axs[0].legend()
    axs[0].set_xticks(bar_positions)
    axs[0].set_xticklabels(keys, rotation=45)

    # Second subplot: Error
    axs[1].bar(bar_positions, model_errors, width=bar_width, color='green')

    axs[1].bar(bar_positions + bar_width, baseline_errors, width=bar_width, color='blue')

    axs[1].set_title('Error')
    axs[1].set_ylabel('Error Value')
    axs[1].set_xticks(bar_positions)
    axs[1].set_xticklabels(keys, rotation=45)
    axs[1].legend(['Model Error','Baseline Error'])


    # Third subplot: Average Model Error vs Average Baseline Error
    avg_model_error = np.mean(model_errors)
    avg_baseline_error = np.mean(baseline_errors)

    bar_width = 0.3

    fig, ax = plt.subplots(figsize=(5, 5))

    # Set position for the single set of bars
    avg_bar_position = np.array([0])  # Center the bars at position 0
    ax.bar(avg_bar_position - bar_width/2, [avg_model_error], width=bar_width, color='green', label='Average Model Error')
    ax.bar(avg_bar_position + bar_width/2, [avg_baseline_error], width=bar_width, color='blue', label='Average Baseline Error')

    # Set title and labels using the ax object
    ax.set_title('Average Error Comparison')
    ax.set_ylabel('Average Error Value')
    ax.set_xticks(avg_bar_position)
    ax.set_xticklabels(['Average Error'])
    ax.legend()



    # Adjust layout
    plt.tight_layout()
    plt.show()