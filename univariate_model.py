from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math

class Net1(nn.Module):

    # A linear regression using a neural network

    def __init__(self):
        super(Net1, self).__init__()

        observation_width = 1
        connect_1_2 = 125
        connect_2_3 = 125
        connect_3_4 = 125
        connect_4_5 = 1

        self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(in_features=connect_3_4, out_features=connect_4_5)


        torch.nn.init.uniform_(self.fc1.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc2.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc3.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc4.weight, -1 * 0.5, 0.5)


    def forward(self, x):

        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out

class Net2(nn.Module):

    # A linear regression using a neural network

    def __init__(self, dropout):
        super(Net2, self).__init__()

        self.dropout = dropout

        observation_width = 1
        connect_1_2 = 32000
        connect_2_3 = 1

        self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)

        torch.nn.init.uniform_(self.fc1.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc2.weight, -1 * 0.5, 0.5)

    def forward(self, x):

        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)

        return out

def curve(x):
    y = math.erf(x-2)*math.sin(100/x)
    return y

def buildBatch(minibatch_size, training_set_definition):

    training_set_x = []
    training_set_y = []
    batch_size = 0

    while batch_size < minibatch_size:
        for item in training_set_definition:
            x = random.uniform(item[0],item[1])
            r1 = random.uniform(0,1)
            if r1 < item[2]:
                y = curve(x)
                y = random.gauss(y,0.02)
                training_set_x.append(torch.tensor(x, requires_grad=True, dtype=float).unsqueeze(0).unsqueeze(0).float())
                training_set_y.append(torch.tensor(y, requires_grad=True, dtype=float).unsqueeze(0).unsqueeze(0).float())
                batch_size += 1

    set_x = torch.cat(training_set_x, 0)
    set_y = torch.cat(training_set_y, 0)

    return set_x, set_y

def train(net_model, optimizer, criterion, minibatch_size, training_set_definition, iterations):
    net_model.train()
    for iteration in range(iterations):
        net_model.zero_grad()
        x, y = buildBatch(minibatch_size,training_set_definition)
        y_pred = net_model(x)
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        print(f'Training Iteration: {iteration}   Loss: {loss.item()}')

def test(net_model, test_set_definition, iterations, batch_size):
    net_model.eval()
    test_results_x = []
    test_results_y = []
    test_results_y_pred = []
    for iteration in range(iterations):
        x, y = buildBatch(batch_size, test_set_definition)
        y_pred = net_model(x)
        for idx in range(len(x)):
            test_results_x.append(x[idx].item())
        for idx in range(len(y)):
            test_results_y.append(y[idx].item())
        for idx in range(len(y_pred)):
            test_results_y_pred.append(y_pred[idx].item())
        print(f'Test Iteration: {iteration}')
    return (test_results_x, test_results_y, test_results_y_pred)

def plot_results(test_results,experiment):
    test_results_x = test_results[0]
    test_results_y = test_results[1]
    test_results_y_pred = test_results[2]
    plt.scatter(test_results_x, test_results_y)
    plt.scatter(test_results_x, test_results_y_pred)
    plt.title(f'Univariate')
    plt.legend(['Test Set', 'Predicted Value'])
    experiment.log_figure(figure=plt)
    plt.show()

def main():
    experiment = Experiment(api_key="1x1ZQpvbtvDyO2s5DrlUyYpzv", project_name="univariate4", workspace="verlyn-fischer")

    # Hyperparameters
    training_iterations = 1000
    test_iterations = 4
    minibatch_size = 512
    learning_rate = 0.001
    network = '32000'
    momentum = 0.5
    dropout = 0.0
    optim_type = 'adam'
    weight_decay = 0.0001

    hyper_params = {'network': network, 'batch_size': minibatch_size,
                    'learning_rate': learning_rate, 'momentum': momentum, 'iterations':training_iterations,'dropout':dropout,'optim':optim_type,'weight_decay':weight_decay}
    experiment.log_parameters(hyper_params)

    # net_model = Net1() # Three Layer
    net_model = Net2(dropout) # Single Layer


    # optimizers
    if optim_type == 'adam':
        optimizer = optim.Adam(net_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net_model.parameters(), lr=learning_rate, momentum=momentum)

    criterion = nn.MSELoss()

    # LEFT
    left_model_file = 'results/model_left_only_fig_1.1.pth'
    # training_set_definition = [(1.4, 2.1, 1.0),(2.1, 2.8, 1.00)]
    # test_set_definition = [(1.4, 2.8, 1.00)]
    # train(net_model, optimizer, criterion, minibatch_size, training_set_definition, training_iterations)
    # torch.save(net_model.state_dict(), left_model_file)
    # test_results = test(net_model, test_set_definition, test_iterations, minibatch_size)
    # plot_results(test_results,experiment)

    # RIGHT
    right_model_file = 'results/model_right_pinning_fig_3.2.pth'
    training_set_definition = [(2.0, 2.1, 0.1),(2.1, 2.8, 1.00)]
    test_set_definition = [(1.4, 2.8, 1.00)]
    net_model.load_state_dict(torch.load(left_model_file))
    train(net_model, optimizer, criterion, minibatch_size, training_set_definition, training_iterations)
    torch.save(net_model.state_dict(), right_model_file)
    test_results = test(net_model, test_set_definition, test_iterations, minibatch_size)
    plot_results(test_results,experiment)

main()


