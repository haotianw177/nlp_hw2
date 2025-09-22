class Vocab:
    def __init__(self):
        self.sym2num = {}
        self.num2sym = []

    def add(self, sym):
        if sym not in self.sym2num:
            self.sym2num[sym] = len(self.num2sym)
            self.num2sym.append(sym)

    def numberize(self, sym):
        return self.sym2num.get(sym, self.sym2num['<UNK>'])

    def denumberize(self, num):
        return self.num2sym[num]

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
