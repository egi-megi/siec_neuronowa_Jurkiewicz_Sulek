import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def create_plot_name(name: str) -> str:
    plots_dir_name = "plots"
    return plots_dir_name + "/" + name


def prepare_data_from_csv(file_name: str):
    results_dir_name = "results"

    file_path = results_dir_name + "/" + file_name
    print("Reading file from: \"" + file_path + "\"")
    results = pd.read_csv(file_path, names=["loss_mean", "loss_stddev", "epochs_mean", "name"])
    # data modification
    results[['temp1', 'temp2', 'neurons_1', 'temp3', 'neurons_2']] = results.name.str.split("_", expand=True)
    activation_function_name = results.temp2[0]
    results['neurons_1'] = pd.to_numeric(results['neurons_1'])
    results['neurons_2'] = pd.to_numeric(results['neurons_2'])
    return results.drop(columns=["temp1", "temp2", "temp3"]), activation_function_name


def produce_plots_2_layers(results_file_name: str):
    results, activation_function_name = prepare_data_from_csv(results_file_name)

    ax = plt.axes(projection='3d')
    ax.set_title(activation_function_name)
    ax.set_xlabel('neurons layer 1')
    ax.set_ylabel('neurons layer 2')
    ax.set_zlabel('loss')
    ax.plot_trisurf(results.neurons_1, results.neurons_2, results.loss_mean, cmap='viridis')
    plt.savefig(create_plot_name(activation_function_name + "_loss_mean"))
    plt.show()

    idx_min = results['loss_mean'].idxmin()
    print_min(results, activation_function_name)
    chosen_no_of_neurons_1 = results['neurons_1'][idx_min]
    data_of_chosen_l1 = results[results['neurons_1'] == chosen_no_of_neurons_1]
    ax = plt.axes()
    ax.set_title(activation_function_name + " l1=" + str(chosen_no_of_neurons_1))
    ax.set_xlabel('neurons layer 2')
    ax.set_ylabel('loss')
    plt.plot(data_of_chosen_l1['neurons_2'], data_of_chosen_l1['loss_mean'], 'bo--')
    plt.savefig(create_plot_name("2_" + activation_function_name + "_loss_mean_for_l1_" + str(chosen_no_of_neurons_1)))
    plt.show()

    chosen_no_of_neurons_2 = results['neurons_2'][idx_min]
    data_of_chosen_l2 = results[results['neurons_2'] == chosen_no_of_neurons_2]
    ax = plt.axes()
    ax.set_title(activation_function_name + " l2=" + str(chosen_no_of_neurons_2))
    ax.set_xlabel('neurons layer 1')
    ax.set_ylabel('loss')
    plt.plot(data_of_chosen_l2['neurons_1'], data_of_chosen_l2['loss_mean'], 'bo--')
    plt.savefig(create_plot_name("2_" + activation_function_name + "_loss_mean_for_l2_" + str(chosen_no_of_neurons_2)))
    plt.show()


def print_min(results, activation_function_name):
    idx_min = results['loss_mean'].idxmin()
    print("minimal loss for " + activation_function_name + ": " + str(results['loss_mean'].min()))
    print("for l1=" + str(results['neurons_1'][idx_min]) + " l2=" + str(results['neurons_2'][idx_min]))


def produce_plot_for_1_layer(results_file_name: str):
    results, activation_function_name = prepare_data_from_csv(results_file_name)
    print_min(results, activation_function_name)
    results = results[results['neurons_1'] <= 360]

    ax = plt.axes()
    ax.set_title("one layer loss function\n activation: " + activation_function_name)
    ax.set_xlabel('neurons layer 1')
    ax.set_ylabel('loss')
    plt.plot(results['neurons_1'], results['loss_mean'], 'bo--')
    plt.savefig(create_plot_name("1_" + activation_function_name + "_loss_mean"))
    plt.show()


if __name__ == '__main__':
    produce_plot_for_1_layer("1_sigmoid_average.csv")
    produce_plot_for_1_layer("1_tanh_average.csv")
    produce_plot_for_1_layer("1_elu_average.csv")
    produce_plot_for_1_layer("1_swish_average.csv")

    produce_plots_2_layers("2_sigmoid_sigmoid_average.csv")
    produce_plots_2_layers("2_tanh_tanh_average.csv")
    produce_plots_2_layers("2_elu_elu_average.csv")
    produce_plots_2_layers("2_swish_swish_average.csv")
