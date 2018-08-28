import numpy as np
import random


# Dataset and preprocessing

data = open('dinos.txt','r').read()
data = data.lower()
chars = list(set(data))

data_size = len(data)
vocab_size = len(chars)

print("total number of character is %d, and total number of unique character is %d ".%(data_size, vocab_size))


# mapping character to index, index to character
char_to_ix = {ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}


def clip(gradients, maxvalue):
  '''
  clip the gradients' values between minimum and maximum
  Arguments:
  gradients -- a dictionary containing the gradients "dwaa","dwax","dwya","db","dby"
  maxvalue -- everything above this number is set to this number, and everything less than -maxvalue is set to -maxvalue
  
  Returns:
  gradients -- a dictionary with the clipped gradients
  '''
  
  dwaa = gradients['dwaa']
  dwax = gradients['dwax']
  dwya = gradients['dwya']
  db = gradients['db']
  dby = gradients['dby']
  
  for gradient in [dwax,dwaa,dwya,db,dby]:
    np.clip(gradient, -maxvalue, maxvalue, out = gradient)
    
  gradients = {"dwaa":dwaa,"dwax":dwax,"dwya":dwya,"db":db,"dby":dby}
  
  return gradients


def sample(parameters, char_to_ix,seed):
  """
  Sample a sequence of characters according to a sequence of probability distributions output of the RNN
  Arguments:
  parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
  char_to_ix -- python dictionary mapping each character to an index.
  seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
  """
  waa = parameters['waa']
  wax = parameters['wax']
  wya = parameters['wya']
  by = parameters['by']
  b = parameters['b']
  
  vocab_size  = by.shape[0]
  n_a = waa.shape[1]
  
  x = np.zeros((vocab_size, 1))
  a_prev = np.zeros((n_a,1))
  
  idices = []
  
  idx = -1
  
  counter = 0
  
  newline_character = char_to_ix['\n']
  
  while (idx != newline_character and counter !=50):
    a = np.tanh(np.dot(wax,x) + np.dot(waa,a_prev) + b)
    z = np.dot(wya,a) + by
    y = softmax(z)
    
    idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
    indices.append(idx)
    
    x = np.zeros((vocab_size,1))
    x[idx] = 1
    
    a_prev = a
    
    seed += 1
    counter +=1
    
  if (counter == 50):
    indices.append(char_to_ix['\n'])
  
  return indices


 
  def optimize(x,y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    # forward propagation through time
    loss, cache = rnn_forward(x,y,a_prev,parameters)
    
    #Back propagate through time 
    gradients, a  = rnn_backward(x,y,parameters, cache)
    
    #clip your gradient  between -5 (min) and  5 (max)
    gradients = clip(gradients,5)
    
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(x)-1]
    
    
    
    def model(data,ix_to_char, char_to_ix,num_iterations = 35000, n_a=50,dino_names=7, vocab_size=27):
      """
      Trains the model and generates dinosaur names. 
      Arguments:
      data -- text corpus
      ix_to_char -- dictionary that maps the index to a character
      char_to_ix -- dictionary that maps a character to an index
      num_iterations -- number of iterations to train the model for
      n_a -- number of units of the RNN cell
      dino_names -- number of dinosaur names you want to sample at each iteration. 
      vocab_size -- number of unique characters found in the text, size of the vocabulary
      Returns:
      parameters -- learned parameters
      """
      n_x,n_y = vocab_size, vocab_size
      
      parameters = initialize_parameters(n_a,n_x, n_y)
      loss = get_initial_loss(vocab_size, dino_names)
      
      with open("dinos.txt") as f:
        examples = f.readlines()
      examples = [x.lower().strip() for x in examples]
      
      np.random.seed(0)
      np.random.shuffle(examples)
      
      a_prev = np.zeros((n_a,1))
      
      for j in  range(num_iterations):
        
        index = j % len(examples)
        x = [None] + [char_to_ix[ch] for ch in examples[index]]
        y = x[1:] + [char_to_ix["\n"]]
        
        curr_loss, gradients, a_prev = optimize(x,y,a_prev,parameters)
        
        loss = smooth(loss,curr_loss)
        
        if j % 2000 == 0:
          print('Iteration: %d, Loss: %f' %(j,loss) + '\n')
          
          seed = o
          for name in range(dino_names):
            sampled_indices = sample(parameters, char_to_ix,seed)
            print_sample(sample_indices,ix_to_char)
            seed +=1
          print('\n')
        
      return parameters  
          
         
                
                
