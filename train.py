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
  
  return parameters
 
    
   
                
                
                
