import torch
import torchvision


class String2Int:
    def __init__(self, labels, start_token, stop_token, pad_token, remove_punct=""):
        """
        Encodes strings as integers and saves useful information about the string-integer mappings.

        labels = list of all labels (in train, valid, test) to be used to generate the vocabulary.\n
        start_token = the special token used to start a phrase, such as <sos>\n
        stop_token = the special token used to stop a phrase, such as <eos> (should not be same as start)\n
        pad_token = the special token used to pad phrases to the same length\n
        remove_punct = a single string listing punctuation to remove from the labels before encoding them\n

        Call with a str to return an int, and call with an int to return a str.
        """
        all_words = ' '.join(labels)
        all_words = ''.join([c for c in all_words if c not in remove_punct])
        unique_words = sorted(set(all_words.lower().split()))
        stoi = {word:i+3 for i,word in enumerate(unique_words)}
        stoi[start_token] = 0
        stoi[stop_token] = 1
        stoi[pad_token] = 2

        self.stoi = stoi
        self.itos = {i:s for s,i in stoi.items()}
        self.start_token=start_token
        self.stop_token=stop_token
        self.pad_token=pad_token
        self.remove_punct=remove_punct
    
    def __len__(self):
        return len(self.stoi)

    def __call__(self, item):
        """Map from string-to-integer or integer-to-string based on type(item)"""
        try:
            if type(item)==str:
                return self.stoi[item]
            elif type(item)==int:
                return self.itos[item]
            else:
                raise ValueError
        except ValueError as e:
            print(f"ValueError: Item is of type {type(item)} and not str or int")


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, X_paths, y_labels, string2int:String2Int, transforms=None): #device?
        """
        X_paths = ordered list of image paths\n
        y_labels = ordered list of image captions\n
        string2int = a String2Int object or a python dict containing 'stoi', 'start_token', 'stop_token', 'pad_token', and 'remove_punct'\n 
        transform = transformation/preprocessing function applied to images\n
        """

        self.X_paths = X_paths
        self.y_labels = y_labels
        self.transforms = transforms
        self.most_words = len(max([lab.split() for lab in y_labels], key=len)) + 2 ##for the start/stop tokens
        
        self.string2int = string2int
        self.start_tok_idx = string2int.stoi[string2int.start_token]
        self.stop_tok_idx = string2int.stoi[string2int.stop_token]
        self.pad_tok_idx = string2int.stoi[string2int.pad_token]
    
    def __len__(self):
        assert len(self.X_paths)==len(self.y_labels)
        return len(self.X_paths)

    def __getitem__(self, idx):
        # IMAGE
        img_path = self.X_paths[idx]
        img = torchvision.io.read_image(img_path)
        img = self._prep_image(img.float(), transforms=self.transforms)

        # LABEL
        str_label = self.y_labels[idx]
        str_label = ''.join([c for c in str_label if c not in self.string2int.remove_punct])
        int_label = [self.start_tok_idx] + [self.string2int.stoi[s] for s in str_label.split()] + [self.stop_tok_idx]

        return img, torch.tensor(self._pad_label(int_label))

    def _prep_image(self, img_tensor, transforms=None):
        img_tensor *= 1.0/255
        if transforms is not None:
            img_tensor = transforms(img_tensor)
        return img_tensor

    def _pad_label(self, int_label):
        int_label += [self.pad_tok_idx] * (self.most_words - len(int_label))
        return int_label