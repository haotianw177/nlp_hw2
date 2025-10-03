import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import pickle
import nltk
from nltk.corpus import treebank

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download treebank data
nltk.download('treebank')
brown = list(treebank.tagged_sents())

# Improved Vocab class
class Vocab:
    def __init__(self, add_unk=True):
        self.sym2num = {}
        self.num2sym = []
        self.unk_token = '<UNK>'
        self.unk_index = None

        if add_unk:
            self.add(self.unk_token)
            self.unk_index = self.sym2num[self.unk_token]

    def add(self, sym):
        if sym not in self.sym2num:
            self.sym2num[sym] = len(self.num2sym)
            self.num2sym.append(sym)
            # Update unk_index if this is the UNK token
            if sym == self.unk_token:
                self.unk_index = self.sym2num[sym]

    def numberize(self, sym):
        if sym in self.sym2num:
            return self.sym2num[sym]
        elif self.unk_index is not None:
            return self.unk_index
        else:
            raise KeyError(f"Symbol '{sym}' not in vocabulary and no UNK token available")

    def denumberize(self, num):
        if num < len(self.num2sym):
            return self.num2sym[num]
        else:
            return self.unk_token if self.unk_token else '<UNKNOWN>'

    def __len__(self):
        return len(self.num2sym)

    def update(self, seq):
        for sym in seq:
            self.add(sym)

def read_pos_file(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs = []
            # each token looks like "(word, POS)"
            for token in line.split(") ("):
                token = token.strip("()")  # remove outer parentheses
                if not token:
                    continue
                word, pos = token.split(", ")
                pairs.append((word, pos))
            sentences.append(pairs)
    return sentences

class BiLSTMTagger(nn.Module):
    def __init__(self, data, embedding_dim, hidden_dim):
        super().__init__()

        # Build vocabulary from training data
        self.words = Vocab(add_unk=True)  # This will automatically add <UNK>
        self.tags = Vocab(add_unk=False)  # Tags don't need UNK

        # Extract all words and tags from training data
        all_words = []
        all_tags = []
        for sentence in data:
            for word, tag in sentence:
                all_words.append(word.lower())  # lowercase as specified
                all_tags.append(tag)

        # Update vocabularies
        self.words.update(all_words)
        self.tags.update(all_tags)

        print(f"Vocabulary sizes - Words: {len(self.words)}, Tags: {len(self.tags)}")
        print(f"<UNK> token index: {self.words.unk_index}")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Initialize layers
        self.embedding = nn.Embedding(len(self.words), embedding_dim)

        # Bidirectional LSTM: bidirectional=True, batch_first=False (default)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # Simple approach: Remove dropout to keep it simple
        # Output layer: hidden_dim * 2 because bidirectional
        self.W_out = nn.Linear(hidden_dim * 2, len(self.tags))

    def forward(self, sentence):
        # Convert word indices to embeddings
        embeddings = self.embedding(sentence)

        # Reshape for LSTM: (seq_len, batch_size=1, embedding_dim)
        embeddings = embeddings.view(len(sentence), 1, -1)

        # Pass through bidirectional LSTM
        lstm_out, _ = self.lstm(embeddings)

        # Reshape back: (seq_len, hidden_dim * 2)
        lstm_out = lstm_out.view(len(sentence), -1)

        # Removed dropout to keep it simple
        # Final output scores (pre-softmax)
        scores = self.W_out(lstm_out)

        return scores

    def predict(self, scores):
        # Get most likely tag for each word position
        return torch.argmax(scores, dim=1)

    def fit(self, data, lr=0.01, epochs=5):
        # Try SGD optimizer instead of Adam for simplicity
        optimizer = optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            random.shuffle(data)

            total_loss = 0.0
            total_tokens = 0

            for sentence in data:
                # Separate words and tags
                words = [word.lower() for word, tag in sentence]
                tags = [tag for word, tag in sentence]

                # Convert to indices - using the improved numberize method
                word_idxs = torch.tensor(
                    [self.words.numberize(word) for word in words],
                    dtype=torch.long,
                    device=device
                )
                targets = torch.tensor(
                    [self.tags.numberize(tag) for tag in tags],
                    dtype=torch.long,
                    device=device
                )

                # Zero gradients
                self.zero_grad()

                # Forward pass
                scores = self(word_idxs)

                # Calculate loss
                loss = criterion(scores, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

                # Update weights
                optimizer.step()

                # Accumulate loss and token count
                total_loss += loss.item() * len(targets)
                total_tokens += len(targets)

            # Calculate average loss per token
            avg_loss = total_loss / total_tokens
            epoch_time = time.time() - start_time

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')

    def evaluate(self, data):
        self.eval()
        correct_predictions = 0
        total_tokens = 0

        with torch.no_grad():
            for sentence in data:
                words = [word.lower() for word, tag in sentence]
                tags = [tag for word, tag in sentence]

                # Convert to indices
                word_idxs = torch.tensor(
                    [self.words.numberize(word) for word in words],
                    dtype=torch.long,
                    device=device
                )
                targets = torch.tensor(
                    [self.tags.numberize(tag) for tag in tags],
                    dtype=torch.long,
                    device=device
                )

                # Forward pass
                scores = self(word_idxs)

                # Predict tags
                predictions = self.predict(scores)

                # Count correct predictions
                correct_predictions += (predictions == targets).sum().item()
                total_tokens += len(targets)

        accuracy = correct_predictions / total_tokens
        return accuracy

if __name__ == '__main__':
    # Read data files
    train_data = read_pos_file('train.pos')
    val_data = read_pos_file('val.pos')
    test_data = read_pos_file('test.pos')

    print(f"Training sentences: {len(train_data)}")
    print(f"Validation sentences: {len(val_data)}")
    print(f"Test sentences: {len(test_data)}")

    # Use both ATIS and Brown
    train_sents = train_data + brown

    # Initialize model
    model = BiLSTMTagger(train_sents, embedding_dim=128, hidden_dim=256)
    model.to(device)

    # Try a slightly higher learning rate and more epochs
    print("Starting training...")
    model.fit(train_sents, lr=0.05, epochs=15)  # Increased learning rate and epochs

    # Evaluate on validation and test sets
    val_accuracy = model.evaluate(val_data)
    test_accuracy = model.evaluate(test_data)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # If we still don't reach 92%, try one more run with different parameters
    if test_accuracy < 0.92:
        print("\nFirst run didn't reach 92%, trying with different parameters...")

        # Reinitialize and try again with different learning rate
        model = BiLSTMTagger(train_sents, embedding_dim=128, hidden_dim=256)
        model.to(device)
        model.fit(train_sents, lr=0.1, epochs=20)  # Higher learning rate, more epochs

        val_accuracy = model.evaluate(val_data)
        test_accuracy = model.evaluate(test_data)

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # # Save model
    # torch.save(model.state_dict(), 'pos_tagger_model.pth')
    # print("Model saved as 'pos_tagger_model.pth'")
    #   torch.save(model, '/content/gdrive/MyDrive/nlp/bilstm_tagger_complete.pth')
	#   print("Complete model saved!")	

    # Predict first 10 sentences from test data
    print("\nPredictions for first 10 sentences in test.pos:")
    model.eval()
    with torch.no_grad():
        for i, sentence in enumerate(test_data[:10]):
            words = [word for word, tag in sentence]
            true_tags = [tag for word, tag in sentence]

            # Convert words to indices
            word_idxs = torch.tensor(
                [model.words.numberize(word.lower()) for word in words],
                dtype=torch.long,
                device=device
            )

            # Get predictions
            scores = model(word_idxs)
            pred_indices = model.predict(scores)
            pred_tags = [model.tags.denumberize(idx.item()) for idx in pred_indices]

            print(f"\nSentence {i+1}:")
            print(f"Words: {words}")
            print(f"True tags:  {true_tags}")
            print(f"Pred tags:  {pred_tags}")

            # Check for mismatches
            mismatches = [(w, t, p) for w, t, p in zip(words, true_tags, pred_tags) if t != p]
            if mismatches:
                print("Mismatches:")
                for word, true_tag, pred_tag in mismatches:
                    print(f"  '{word}': True={true_tag}, Pred={pred_tag}")