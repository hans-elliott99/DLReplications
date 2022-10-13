


class RunningMean:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0
        self.sum = 0
        self.mean = 0

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.sum += value*n
        self.mean = self.sum / self.count

class MetricLog:
    def __init__(self, filepath, reset=True) -> None:
        self.file = filepath
        if reset:
            self.reset()

    def reset(self):
        open(self.file, 'w').close() #empty the log file if it exists

    def update(self, epoch, value):
        """Append to log file"""
        with open(self.file, 'a') as f:
            f.write(f'{epoch} {value}\n')

def LoadMetricLog(filepath):
    values = []
    with open(filepath, "r") as f:
        for line in f:
            value = line.strip().split()[-1]
            values.append( float(value) )
    return values

def accuracy(logits, targets, topk=1):
    """
    Compute the (top k) accuracy score given the (unpacked) logits and targets.
    """
    batch_size = logits.size(0)
    _, idx = logits.topk(k=topk, dim=1, largest=True, sorted=True)

    correct = idx.eq( targets.view(-1, 1) ) ##bool vector, True where predicted token==correct token
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)