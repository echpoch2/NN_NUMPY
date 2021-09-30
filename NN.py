
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    def one_hot(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot(data[:,:1])

X_train, y_train = load_data('mnist_train.csv')
X_test, y_test = load_data('mnist_test.csv')

class NeuralNetwork:
    def __init__(self, X, y, batch = 64, lr = 1e-3,  epochs = 50):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.sv_coef =.75
        self.x = self.input[:self.batch] # batch input 
        self.y = self.target[:self.batch] # batch target value
        self.loss = []
        self.loss_test = []
        self.acc = []
        
        self.init_weights()
      
    def init_weights(self):
        #self.W1 = np.random.randn(self.input.shape[1],256)
        #self.W2 = np.random.randn(self.W1.shape[1],128)

        self.W2 = np.random.randn(self.input.shape[1],128)
        self.W3 = np.random.randn(self.W2.shape[1],self.y.shape[1])

        self.W2_hist = np.zeros(self.W2.shape)
        self.W3_hist = np.zeros(self.W3.shape)

        self.b2 = np.random.randn(self.W2.shape[1],)
        self.b3 = np.random.randn(self.W3.shape[1],)

        self.b2_hist = np.zeros(self.W2.shape[1],)
        self.b3_hist = np.zeros(self.W3.shape[1],)

    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 
    
    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
    
    def cross_entropy(self, Y_hat, Y):
        Y[Y == 0] = 0.0000000000000000001
        Y_hat[Y_hat == 0] = 0.0000000000000000001
        return -np.sum(Y * np.log(Y_hat))
        
    def feedforward(self, test=False):
        #assert self.x.shape[1] == self.W1.shape[0]
        #self.z1 = self.x.dot(self.W1) + self.b1
        #self.a1 = self.ReLU(self.z1)

        #assert self.a1.shape[1] == self.W2.shape[0]
        #self.z2 = self.a1.dot(self.W2) + self.b2
        #self.a2 = self.ReLU(self.z2)



        self.z2 = self.x.dot(self.W2) + self.b2
        self.a2 = self.ReLU(self.z2)


        assert self.a2.shape[1] == self.W3.shape[0]
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        self.error = self.cross_entropy(self.a3, self.y)/self.batch

        
    def backprop(self):
        self.W3 = self.W3 - self.sv_coef * self.W3_hist
        self.W2 = self.W2 - self.sv_coef * self.W2_hist
        self.b3 = self.b3 - self.sv_coef * self.b3_hist
        self.b2 = self.b2 - self.sv_coef * self.b2_hist


        dcost = (1/self.batch)*(self.a3 - self.y)
        DW3 = np.dot(dcost.T,self.a2).T
        DW2 = np.dot((np.dot((dcost),self.W3.T) * self.dReLU(self.z2)).T,self.x).T


        db3 = np.sum(dcost,axis = 0)
        db2 = np.sum(np.dot((dcost),self.W3.T) * self.dReLU(self.z2),axis = 0)

        assert DW3.shape == self.W3.shape
        assert DW2.shape == self.W2.shape

        
        assert db3.shape == self.b3.shape
        assert db2.shape == self.b2.shape

        
        self.W3_hist = self.W3_hist * self.sv_coef + self.lr*DW3
        self.W2_hist = self.W2_hist * self.sv_coef + self.lr*DW2
        self.b3_hist = self.b3_hist * self.sv_coef + self.lr*db3
        self.b2_hist = self.b2_hist * self.sv_coef + self.lr*db2

        self.W3 = self.W3 - self.W3_hist
        self.W2 = self.W2 - self.W2_hist
        self.b3 = self.b3 - self.b3_hist
        self.b2 = self.b2 - self.b2_hist


    def train(self):
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()
            
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end]
                self.y = self.target[start:end]
                self.feedforward()
                self.backprop()
                l+=self.error
                acc+= np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.batch
            print(l)
            self.loss.append(l)
            self.feedforward()
            self.acc.append(acc*100/(self.input.shape[0]//self.batch))
            
    def plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    
    def acc_plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.acc)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        
    def test(self,xtest,ytest):
        self.x = xtest
        self.y = ytest
        self.feedforward()
        acc = np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
        print("Accuracy:", 100 * acc, "%")
    
        
        
NN = NeuralNetwork(X_train, y_train) 
NN.train()
NN.plot()
NN.test(X_test,y_test)