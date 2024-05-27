from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from functions import *
import numpy as np
import math
from sklearn.datasets import load_iris
import random
from sklearn import preprocessing

#import random
random_state=None
seed = None if random_state is None else int(random_state)
rng = np.random.default_rng(seed=seed)

with open("dataset/shakespeare.txt") as data:
    text_data = data.read()[0:500].lower()

print("data: ", text_data)
tokens = word_tokenize(text_data)

# Create a set to get unique words (vocabulary)
vocabulary = list(set(tokens))
#print("vocabulary", vocabulary)
print("size vocabulary: ", len(vocabulary))
print("size vector: ", len(vocabulary[0]))
# Generate one-hot encoded vectors for each word in the vocabulary
one_hot_encoded = []
for word in tokens:
    # Create a list of zeros with the length of the vocabulary
    encoding = [0] * len(tokens)
    
    # Get the index of the word in the vocabulary
    index = list(tokens).index(word)
    
    # Set the value at the index to 1 to indicate word presence
    encoding[index] = 1.0
    one_hot_encoded.append((word, encoding))

#print("one-hot encoding len: ", len(one_hot_encoded))
#print("one-hot encoding[3]: ", one_hot_encoded[3])

class NeuralNetworkRecurrent:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):
        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden #np.random.randn(self.num_neuron_hidden, ) * 0.01
        self.recurcive_hidden = 0
        self.num_times = 0


        self.grad_w2_array = [] 
        self.grad_b2_array = []
        self.grad_recur_array = []
        self.grad_w1_array = []
        self.grad_b1_array = []
        print("--------------------------------------------------")
        print("Neural network: ")
        print("--------------------------------------------------")
        print("Input: ", self.size_input)
        print("Hidden: ", self.num_neuron_hidden)
        print("Output: ", self.size_output)

        self.w1 = np.random.randn(self.num_neuron_hidden, size_input) * 0.01
        print("w1: ", self.w1)
        self.w_recur = np.random.randn(self.num_neuron_hidden, ) * 0.01
        print("w_recur: ", self.w_recur)

        self.w2 = np.random.randn(self.size_output, self.num_neuron_hidden).astype('float64') * 0.01 
        print("w2: ", self.w2)

        self.b1 = np.zeros((self.num_neuron_hidden))
        self.b2 = np.zeros((self.size_output))
        print("--------------------------------------------------")


    def feedforward(self, x):
        self.input_data = x
        #print("self.w1.shape", self.w1.shape)
        #print("x", x)
        #self.num_feed_forward = self.num_feed_forward + 1
        #print("self.num_feed_forward: ", self.num_feed_forward)
        #print("self.w1.T; ", self.w1.T.shape)
        #print("x: ", x[0].shape)
        #print("self.recurcive_hidden: ", self.recurcive_hidden)
        self.z1 = (x @ self.w1.T) #+ (self.recurcive_hidden @ self.w_recur.T) + self.b1
        if self.recurcive_hidden is not 0:
            self.z1 += (self.recurcive_hidden @ self.w_recur.T) 
        else:
            self.z1 += self.b1

        #
        
        self.sigmoid_hidden = sigmoid(self.z1)
        #print("self.sigmoid_hidden: ", self.sigmoid_hidden)
        #print("self.w2.T; ", self.w2.T.shape)
        #print("self.sigmoid_hidden: ", self.sigmoid_hidden.shape)
        self.z2 = (self.sigmoid_hidden @ self.w2.T) + self.b2
        self.sigmoid_output = softmax(self.z2)
        
        self.recurcive_hidden = self.sigmoid_hidden
        self.num_times += 1 
        
        return self.sigmoid_output 


    def backpropogation(self, x, y, i):
        delta = rnn_loss_derivative(y, x) #* deriv_sigmoid(self.z2)         
        #print("delta: ", delta)
        grad_w2 = delta  

        grad_b2 = delta
        self.grad_b2_array.append(grad_b2)
        grad_b2 = np.sum(self.grad_b2_array, axis=0)

        grad_w2 = np.outer(grad_w2, self.sigmoid_hidden)
        self.grad_w2_array.append(grad_w2)
        grad_w2 = np.sum(self.grad_w2_array, axis=0)

        self.w2 = self.w2
        self.w2 -= self.learn_rate * grad_w2.astype('float64') 
        self.b2 -= self.learn_rate * grad_b2.astype('float64')
        
        self.grad_recur = (delta @ self.w2) * deriv_sigmoid(self.z1) * self.recurcive_hidden
        self.grad_recur_array.append(self.grad_recur)
        self.grad_recur = np.sum(self.grad_recur_array, axis=0)

        self.w_recur -= self.grad_recur.astype('float64') 

        delta_input = (delta @ self.w2) * deriv_sigmoid(self.z1)
        grad_w1 = np.outer(delta_input, self.input_data)
        self.grad_w1_array.append(grad_w1)
        grad_w1 = np.sum(self.grad_w1_array, axis=0)
        
        grad_b1 = delta_input.astype('float64')
        self.grad_b1_array.append(grad_b1) 
        grad_b1 = np.sum(self.grad_b1_array, axis=0)
        
        self.w1 -= self.learn_rate * grad_w1.astype('float64') 
        self.b1 -= self.learn_rate * grad_b1.astype('float64')

    def train(self, x, y, all_train):
        #print("all: ", all_train)
        size_data = len(x)
        all_pred = []
        batch_size = 16
        #print("all: ", all_train[:batch])
        #print("num: iter: ", round(size_data / batch))
        num_batch = round(size_data / batch_size) 
        #print("all each 1: ", all_train[num_batch * 1:batch])
        #print("all each 2: ", all_train[num_batch * 2:batch])


        for ep in range(self.epoch):
            rng.shuffle(all_train)
            for index in range(num_batch):
                stop = index + batch_size

                x_batch, y_batch = all_train[index:stop, :-1], all_train[index:stop, -1:]
                for i in range(len(x_batch)):
                    #print("x_batch: ", x_batch[i][0][0], "y_batch: ", y_batch[i][0][0])
                    pred = self.feedforward(x_batch[i][0][0])
                #print("pred: ", pred)
                #print("y: ", y[index])
                    all_pred.append(np.array(pred))
                    self.backpropogation(pred, y_batch[i][0][0], index)
                    error = rnn_loss(pred, y_batch[i][0][0]) 
                      


            all_pred = []
            #print("self.w2 ", self.w2)


            #if ep % 10 == 0:
            print("--------------------")
            print("epoch: ", ep)
            print("times: ", self.num_times)
            print("error", error) 
                #print("setosa: ", network.feedforward(np.array([5.1,3.5,1.4,0.2])))
                #print("setosa argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.1,3.5,1.4,0.2])))))
                #print("versicolor argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.5,2.5,4.0,1.3])))))
                #print("versicolor: ", network.feedforward(np.array([5.5,2.5,4.0,1.3])))
                #print("virginica argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.9,3.0,5.1,1.8])))))
                #print("virginica: ", network.feedforward(np.array([5.9,3.0,5.1,1.8])))



all_train = []
X = []
Y = []

#print("(one_hot_encoded[1][0]: ", one_hot_encoded[1][1])
for i in range(len(tokens)):

    #network.train(text_data[i], text_data[i+1], np.array(all_train))
    try:
        X.append(one_hot_encoded[i][1])
        Y.append(one_hot_encoded[i+1][1])
        #print("X: ", one_hot_encoded[i][1])
        #print("Y: ", one_hot_encoded[i+1][1])
        #print("argmax one_hot_encoded[i][1]: ", np.argmax(one_hot_encoded[i][1]))
        #print("argmax one_hot_encoded[i+1][1]: ", np.argmax(one_hot_encoded[i+1][1]))

        elem = [[one_hot_encoded[i][1]],[one_hot_encoded[i+1][1]]]
        #print("elem: ", elem)
    except IndexError:
        break
        print("end processing")
    all_train.append(np.array(elem, dtype=object))

network = NeuralNetworkRecurrent(0.1, 3, len(one_hot_encoded), 20, len(one_hot_encoded))

try:
    #print("len(all_train) : ", len(all_train))
    network.train(X, Y, np.array(all_train))
except IndexError:
    print("end learning")


size_gen = 3
text = ""
text_data_pred = None
start_word = one_hot_encoded[3][1]
#print("size start_word: ", len(start_word))
two_word = network.feedforward(start_word)
#print("two_word: ", two_word)
three_word = network.feedforward(two_word)
#print("three_word: ", three_word)

text = tokens[np.argmax(start_word)] + " " + tokens[np.argmax(two_word)] + " " + tokens[np.argmax(three_word)] 
print("generate: ", text)

#for word in range(size_gen):




