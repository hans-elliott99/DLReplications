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
    
    def log(self, epoch, value):
        """Append to log file"""
        with open(self.file, 'a+') as f:
            f.write(f"{epoch}; {value}\n")

    def reset(self):
        """Empty the log file if it already exists."""
        open(self.file, 'w').close()

def load_MetricLog(filepath):
    values = []
    with open(filepath, "r") as f:
        for line in f:
            value = line.strip().split()[-1]
            values.append( float(value) )
    return values


def topkaccuracy(logits, targets, topk=1):
    """Compute the (top k) accuracy score given the (unpacked) logits and targets.
    """
    batch_size = logits.size(0)
    _, idx = logits.topk(k=topk, dim=1, largest=True, sorted=True)

    correct = idx.eq( targets.view(-1, 1) ) ##bool vector, True where predicted token==correct token
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)




def adjust_lr_step(optimizer, reduction_factor):
    """Adjust LR for the given optimizer and return new LR value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * reduction_factor
    
    return optimizer.param_groups[0]['lr']

def adjust_lr_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    if ( lr < 1.0e-7 ):
      lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
    