import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import pickle
import nltk
from nltk.corpus import treebank
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('treebank')
brown = list(treebank.tagged_sents())

class BiLSTMTagger(nn.Module):
	def __init__(self, data, embedding_dim, hidden_dim):
		super().__init__()
		self.words = None
		self.tags = None
		"""TODO: Populate self.words and self.tags as two vocabulary objects using the Vocab class in utils.py.
		This will allow you to easily numberize and denumberize the word vocabulary as well as the tagset.
		Make sure to add <UNK> self.words."""
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		"""	TODO: Initialize layers."""
		# embedding
		# lstm
		# dropout
		# W_out
		raise NotImplementedError

	def forward(self, sentence):
		"""	TODO: Pass the sentence through the layers of the model. 
			* Because we are using the built-in LSTM, we can pass in an entire sentence rather than iterating through the tokens.
			* IMPORTANT: Because we are dealing with a full sentence now, we have to do minor reshaping. 
				* Before passing the embeddings into the LSTM, we have to do `embeddings.view(len(sentence), 1, -1)`
				* Before passing the LSTM output into dropout, we have to do `lstm_out.view(len(sentence), -1)`
			* Return the output scores from the model (pre-softmax). This will be of shape: len(sentence) x total number of tags, meaning each row corresponds to a word, and the values in each row are the scores for all possible POS tags for that word."""
		raise NotImplementedError

	def predict(self, scores):
		"""	TODO: Return the most likely tag sequence.
			* When the dim argument is provided, torch.argmax(input, dim) returns a tensor containing the indices of the maximum values along that specified dimension.
			* Since each row of scores corresponds to a different word, and each column corresponds to a different tag, specificy dim=1 (take max along columns)."""
		raise NotImplementedError

	def fit(self, data, lr=0.01, epochs=5):
		"""	TODO: This function is very similar to fit() from HW1."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		
		# 2. Set a loss function variable to `nn.CrossEntropyLoss()`. It includes softmax.
		
		# 3. Loop through the specified number of epochs.
		
		#	 1. Put the model into training mode using `self.train()`.
		
		#	 2. Shuffle the training data using random.shuffle().
		
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of tokens (`total_tokens`).
		
		#	 4. Loop over each sentence in the training data.
		
		#	 	1. Produce a numberized sequence of the words in the sentence. Make words lowercase first, and convert the sequence to a tensor using something like: `torch.tensor(idxs, dtype=torch.long)`.

		#		2. Prepare the target labels using something like: `targets = torch.tensor([self.tags.numberize(t) for t in tags], dtype=torch.long)`
		
		#	 	3. Call `self.zero_grad()` to clear any accumulated gradients from the previous update.
		
		#	 	4. Pass the prepared sequence into the model by doing `self(sentence)` to obtain scores. This automatically calls forward().

		#		5. Calculate loss, passing in the output scores and the true target labels.
		
		#	 	6. Call `loss.backward()` to compute gradients.

		#		7. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.

		#		8. Call `optimizer.step()` to update the model parameters using the computed gradients.

		#		9. Add `loss.item() * len(targets)` to `total_loss`.

		#		10. Add `len(targets)` to `total_tokens`.

		#	5. Compute the average loss per token by dividing `total_loss / total_tokens`.

		#	6. For debugging, it will be helpful to print the average loss per token and the runtime after each epoch. Average loss per token should always decrease epoch to epoch.

		raise NotImplementedError

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate POS tagging accuracy. 
			* Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			* Prepare the sequence and target labels as in fit().
			* Use self.predict() to get the predicted tags, and then check if it matches the real next character found in the data.
			* Divide the total correct predictions by the total number of tokens to get the final accuracy."""
		raise NotImplementedError

if __name__ == '__main__':
	"""TODO: (reference HW1 part3.py)
	* Use read_pos_file() from utils.py to read train.pos, val.pos, and test.pos.
	* Initialize the model with training data, embedding dim 128, and hidden dim 256.
	* Train the model, calling fit(), on the training data.
	* Test the model, calling evaluate(), on the validation and test data.
	* Predict outputs for the examples in response.txt.
	* Remove all instances of `raise NotImplementedError`!
	"""
	raise NotImplementedError
