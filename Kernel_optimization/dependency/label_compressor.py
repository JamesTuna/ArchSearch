class LabelCompressor:
    def __init__(self):
        self.counter = 0
        self.label_set = {}

    def compress(self, label):
        if label in self.label_set:
            return self.label_set[label]
        else:
            self.counter += 1
            self.label_set[label] = self.counter
            return self.counter
