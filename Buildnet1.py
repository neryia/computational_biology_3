import math
import random
import numpy as np
import matplotlib.pyplot as plt


global X_train, y_train, X_test, y_test
global x_p, y_p
mut_rate = 0.5
a = []
b = []


# Define the activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    if x > 30:
        return 1
    if x < -30:
        return 0
    return 1 / (1 + np.exp(-x))


# Define the model architecture
class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


# Define the Dense layer
class Dense:
    def __init__(self, units, activation=None, weights=None, biases=None):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        z = x @ self.weights + self.biases  # Use matrix multiplication
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        else:
            return z


def get_chosen_model(random_number, cumulative_odds, models_list):
    """Returns the model with that chosen."""
    for i, odds in enumerate(cumulative_odds):
        if random_number <= odds:
            return models_list[i]


def combine(chosen1, chosen2):
    w = []
    for j in range(6):
        w.append(np.array([]))
        index = random.choice([x for x in range(chosen1[j].shape[0])])
        for i in range(chosen1[j].shape[0]):
            o = np.array([])
            if j < 3:
                length = len(chosen1[j][i])
            else:
                length = 1
            for k in range(length):
                if k < index:
                    if j < 3:
                        o = np.append(o, chosen1[j][i, k])
                    else:
                        o = np.append(o, chosen1[j][k])
                else:
                    if j < 3:
                        o = np.append(o, chosen2[j][i, k])
                    else:
                        o = np.append(o, chosen2[j][k])
            w[j] = np.append(w[j], o)
        w[j] = w[j].reshape(chosen1[j].shape)
    return w


def choose(model_list, num):
    """Chooses two models from a list."""
    list_of_best = []
    odds = [math.pow(model[6], 4) for model in model_list]
    sum_odds = sum(odds)
    cumulative_odds = np.cumsum(odds) / sum_odds
    for _ in range(num - 5):
        random_number = random.random()
        random_number2 = random.random()
        chosen_model = get_chosen_model(random_number, cumulative_odds, model_list)
        chosen_model2 = get_chosen_model(random_number2, cumulative_odds, model_list)
        list_of_best.append(combine(chosen_model, chosen_model2))
    return list_of_best


def mutate(list_of_best):
    global mut_rate
    ave = 0
    sum = 0
    """Makes random changes to the model."""
    for i, model in enumerate(list_of_best):
        temp = []
        for j, layer in enumerate(model):
            shape = layer.shape
            mutation_rate = random.uniform(0, mut_rate)
            mutation = np.random.uniform(-mutation_rate, mutation_rate, shape)  # Use NumPy broadcasting
            temp.append(np.add(layer, mutation))
        a = evaluate(list_of_best[i])
        b = evaluate(temp)
        if b > a:
            temp.append(b)
            list_of_best[i] = temp
            ave += b-a
            sum += 1
        else:
            list_of_best[i] = tuple(list(list_of_best[i]) + [a])
    if sum < 10:
        mut_rate *= 0.8
    return list_of_best


def lem(a, num):
    a = list(a)
    for k in range(6):
        for i in range(0, num):
            updated_array = []
            for _ in a:
                updated_array.append(np.copy(_))
            # Generate random indices for row and column
            row_index = np.random.randint(updated_array[k].shape[0])
            if k < 3:
                col_index = np.random.randint(updated_array[k].shape[1])
                # Change the random element
                updated_array[k][row_index, col_index] = np.random.randn()
            else:
                updated_array[k][row_index] = np.random.randn()
            temp = evaluate(updated_array)
            if temp > a[6]:
                a = updated_array
                a[6] = temp
    return tuple(a)


def run_generation(model_list):
    """Executes a generation."""
    mutated = mutate(model_list)
    sort_models = sorted(mutated, key=lambda x: x[6], reverse=True)
    print("best - ", sort_models[0][6])
    a.append(sort_models[0][6])
    ave = round(sum([model[6] for model in model_list])/len(model_list), 5)
    print("average - ", ave)
    b.append(ave)
    bests = [lem(lst, 10)[:6] for lst in sort_models[:5]]
    chosen = choose(sort_models[:int(len(model_list)/2)], len(model_list))
    model_list = bests + chosen
    return model_list


def evaluate(weights, tes=False):
    global X_train, y_train
    global X_test, y_test
    # Create the model
    model = Sequential()
    biases1 = weights[3]
    model.add(Dense(64, activation='relu', weights=weights[0], biases=biases1))
    biases2 = weights[4]
    model.add(Dense(64, activation='relu', weights=weights[1], biases=biases2))
    biases3 = weights[5]
    model.add(Dense(1, activation='sigmoid', weights=weights[2], biases=biases3))
    sum = 0
    if tes:
        for x in range(0, len(X_test)):
            output = model.forward(X_test[x])
            sum += float(abs(y_test[x] - output))
        return 1 - sum / len(X_test)
    else:
        for x in range(0, len(x_p)):
            output = model.forward(x_p[x])
            sum += float(abs(y_p[x] - output))
        return 1 - sum / len(x_p)


def percent(p):
    global X_train, y_train, x_p, y_p
    # Determine the length of the arrays
    array_length = len(X_train)

    # Calculate 1% of the length
    selection_count = int(array_length * p)

    # Generate random indices within the range of the array length
    random_indices = random.sample(range(array_length), selection_count)

    # Retrieve the numbers at the randomly selected indices from each array
    selected_numbers1 = [X_train[i] for i in random_indices]
    selected_numbers2 = [y_train[i] for i in random_indices]

    # Create new arrays with the selected numbers
    x_p = selected_numbers1
    y_p = selected_numbers2



def train(num_generations, num_in_each):
    """Trains the model for the specified number of generations."""
    model_list = [[np.random.uniform(-1, 1, (16, 64)),
                   np.random.uniform(-1, 1, (64, 64)),
                   np.random.uniform(-1, 1, (64, 1)),
                   np.random.uniform(-1, 1, 64),
                   np.random.uniform(-1, 1, 64),
                   np.zeros(1),
                   ] for _ in range(num_in_each)]

    for _ in range(num_generations):
        percent(0.01)
        print("round ", _)
        model_list = run_generation(model_list)
    temp = mutate(model_list)
    return sorted(temp, key=lambda x: x[6], reverse=True)[0]


def load(name):
    # Load the data from the file
    data = np.loadtxt(name, delimiter=' ', dtype=str)

    # Split the data into input features and target labels
    X = data[:, 0]
    y = data[:, 3]

    # Data encoding from Strings to int
    X_encoded = np.array([list(map(int, x)) for x in X])
    y_encoded = np.array(list(map(int, y)))

    # Split the data into training and validation sets
    split_ratio = 0.75
    split_index = int(len(X_encoded) * split_ratio)
    global X_train, y_train, X_test, y_test
    X_train = X_encoded[:split_index]
    y_train = y_encoded[:split_index]
    X_test = X_encoded[split_index:]
    y_test = y_encoded[split_index:]


def plot():
    global a, b
    round_numbers = list(range(len(a)))
    plt.plot(round_numbers, a, label='Best')
    plt.plot(round_numbers, b, label='Average')
    plt.xlabel('Round Number')
    plt.ylabel('Score')
    plt.title('Scores vs. Round Number')
    plt.legend()
    plt.grid(True)
    plt.show()


def test(model_):
    print(evaluate(model_, True))


def to_txt(model_):
    temp = []
    for i, n in enumerate(model_):
        if i == 6:
            temp.append(n)
        else:
            for j in n:
                if i < 3:
                    for k in j:
                        temp.append(k)
                else:
                    temp.append(j)
    print(temp)
    my_list_strings = [str(item) for item in temp]

    # Open a new file in write mode
    with open('wnet1.txt', 'w') as file:
        # Write each element of the list to a new line in the file
        for item in my_list_strings:
            file.write(item + '\n')


if __name__ == '__main__':
    load("nn1.txt")
    model = train(200, 200)
    test(model)
    to_txt(model)
    plot()

