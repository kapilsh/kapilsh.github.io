---
title: LSTM Writes Animal Farm
description: >-
  Train LSTM on Animal Farm and create new text
date: 2021-04-21
categories: [Blog, Tutorial]
tags: [Python, Machine Learning, AI, RNN]
pin: true
author: ks
---

RNNs, and specially LSTMs are excellent models for language modelling. In this post, I will train an LSTM character by character to generate sample text from the famous novel [Animal Farm](https://en.wikipedia.org/wiki/Animal_Farm) by George Orwell.

![lstm](/assets/lstm_wide.png)

Image Credits: [https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)

## Setup
Before we start training, we need to load in the animal farm text and create a dataset that is loadable into a PyTorch model. The full animal farm text is available [here](https://github.com/kapilsh/ml-projects/blob/master/rnn/data/animal_farm.txt).

```python
def read_text_file(file_path: str) -> str:
    with open(file_path, 'r') as f:
        text = f.read()
    return text
file_path = ./data/animal_farm.txt
text = read_text_file(file_path)
print(text[:100])
# 'Chapter I\n\nMr. Jones, of the Manor Farm, had locked the hen-houses for the night, but\nwas too 
```

Text in itself is hard to feed into a machine learning model, so typically it is [one hot encoded](https://en.wikipedia.org/wiki/One-hot) into sparse vectors of `[0, 1]` where numbers represent whether individual characters/words are present.

Since we are training char by char, we will tokenize each character as a number. Below I write a function to load all the text and tokenize it:

```python
Tokens = namedtuple("Tokens", ["int_to_chars", "chars_to_int", "tokens"])

def tokenize(text: str) -> Tokens:
    chars = set(text)
    int_to_chars = dict(enumerate(chars))
    chars_to_int = {char: i for i, char in int_to_chars.items()}
    tokens = np.array([chars_to_int[char] for char in text])
    return Tokens(int_to_chars=int_to_chars, chars_to_int=chars_to_int,
                  tokens=tokens)
tokens = tokenize(text)
print(tokens.tokens)
# array([26, 22, 15, 23, 24, 32, 27, 50, 33, 16, 16, 57, 27, 10, 50, 67, 39,
#        7, 32, 47, 51, 50, 39, 68, 50, 24, 22, 32, 50, 57, 15,  7, 39, 27,
#       50, 14, 15, 27, 64, 51, 50, 22, 15, 63, 50, 49, 39, 25, 29, 32, 63,
#       50, 24, 22, 32, 50, 22, 32,  7, 12, 22, 39, 18, 47, 32, 47, 50, 68,
#       39, 27, 50, 24, 22, 32, 50,  7, 56, 52, 22, 24, 51, 50, 42, 18, 24,
#       16, 55, 15, 47, 50, 24, 39, 39, 50, 63, 27, 18,  7, 29, 50])
```

## One Hot Encoding

Now, let's write a method to one-hot encode our data. We will use this method later to encode batches of data being fed into our RNN.

```python
def one_hot_encode(tokens: np.array, label_counts: int) -> np.array:
    result = np.zeros((tokens.size, label_counts), dtype=np.float32)
    result[np.arange(result.shape[0]), tokens.flatten()] = 1
    result = result.reshape((*tokens.shape, label_counts))
    return result
```

## Mini-batching
Since we want to pass the data into our network in mini-batches, next step in the pre-processing is to generate batches od data. In RNNs, we need to pass sequences as the mini-batches. Hence, one way to batch is to split the full sequence into multiple sequences and then grab a window of respective indices from all batches to feed into network.

For example, if the original sequence is of length 20, we can split that into 4 batches of length 5 each. If our window size is 3, we can grab first 3 indices from the 4 batches to pass into the network. Let's look at some code to do it.

```python
def generate_batches(
        sequence: np.array, batch_size: int,
        window: int) -> Generator[Tuple[np.array, np.array], None, None]:
    batch_length = batch_size * window
    batch_count = len(sequence) // batch_length

    truncated_size = batch_count * batch_length
    _sequence = sequence[:truncated_size]
    _sequence = _sequence.reshape((batch_size, -1))

    for n in range(0, _sequence.shape[1], window):
        x = _sequence[:, n:n + window]
        y = np.zeros_like(x)
        if n < _sequence.shape[1]:
            y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, n + window]
        else:
            y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, 0]
        yield x, y

```

Let's test the mini-batch implementation:

```python
batches = generate_batches(tokens.tokens, 10, 40)
x, y = next(batches)
print(x[:5, :6])
print(y[:5, :5])
array([[32, 61, 29, 28, 62, 10],
#       [50, 24, 20, 34, 57, 51],
#       [24, 29, 30, 57, 29, 24],
#       [19, 45, 64,  5, 29, 20],
#       [62, 61, 10, 57, 62, 61]])
#
#array([[61, 29, 28, 62, 10],
#       [24, 20, 34, 57, 51],
#       [29, 30, 57, 29, 24],
#       [45, 64,  5, 29, 20],
#       [61, 10, 57, 62, 61]])
```

## Long Short Term Memory (LSTM) Network

Next, we will define the LSTM model for our training. PyTorch provide a pre-built module for LSTM so we can use that directly. After that we add a dropout layer for regularization followed by a fully connected layer to receive model output. We also need to define what our initial hidden and cell state will be. Let's implement the model class:

```python
class LSTMModel(nn.Module):
    def __init__(self, tokens_size, **kwargs):
        super().__init__()
        self._drop_prob = kwargs.pop("drop_prob")
        self._hidden_size = kwargs.pop("hidden_size")
        self._num_layers = kwargs.pop("num_layers")

        self.lstm = nn.LSTM(
            input_size=tokens_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._drop_prob, batch_first=True)

        self.dropout = nn.Dropout(self._drop_prob)
        self.fc = nn.Linear(self._hidden_size, tokens_size)

    def forward(self, x, h, c):
        x_next, (hn, cn) = self.lstm(x, (h, c))
        x_dropout = self.dropout(x_next)
        x_stacked = x_dropout.contiguous().view(h.shape[1], -1,
                                                self._hidden_size)
        output = self.fc(x_stacked)
        return output, hn, cn

    def initial_hidden_state(self, batch_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self._num_layers, batch_size,
                         self._hidden_size).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self._num_layers, batch_size,
                         self._hidden_size).requires_grad_()

        return h0, c0
```


Now let's test the model by passing in a single batch of data.

```python
data_loader = DataLoader("rnn/data/animal_farm.txt")
tokens = data_loader.tokens
batches = DataLoader.generate_batches(tokens.tokens, 10, 40)
x, y = next(batches)
x = DataLoader.one_hot_encode(x, n_chars)
y = DataLoader.one_hot_encode(y, n_chars)
inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
print(inputs.shape)
print(targets.shape)

model = LSTMModel(len(tokens.int_to_chars), drop_prob=0.1, num_layers=2, hidden_size=256)
h0, c0 = model.initial_hidden_state(batch_size)
output, hn, cn = model(inputs, h0, c0)
print(output.shape)
# torch.Size([10, 40, 71])
# torch.Size([10, 40, 71])
# torch.Size([10, 40, 71])
```

Great! All the dimensions match. Now we can create a training routine to start training and validating our model. I save the different checkpoints of the model after each epoch to compare how the model improves after each epoch. During training, I also use SummaryWriter class from PyTorch that allows us to load results into [Tensorboard](https://www.tensorflow.org/tensorboard).

### ModelRunner

```python
class ModelRunner:
    def __init__(self, data_loader: DataLoader, save_path: str):
        self._data_loader = data_loader
        self._save_path = save_path
        self._tb_writer = SummaryWriter()

    def train(self, parameters: ModelHyperParameters):
        use_gpu = parameters.use_gpu and torch.cuda.is_available()
        if use_gpu:
            logger.info("GPU Available and Enabled: Using CUDA")
        else:
            logger.info("GPU Disabled: Using CPU")

        # load the tokens from the text
        tokens = self._data_loader.tokens

        # define the model
        model = LSTMModel(tokens=tokens,
                          drop_prob=parameters.drop_prob,
                          num_layers=parameters.num_layers,
                          hidden_size=parameters.hidden_size)

        # enable training mode
        model.train()

        # use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parameters.learning_rate)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # split data into training and validation sets
        train_data, valid_data = self._split_train_validation(
            tokens.tokens, parameters.validation_split)

        if use_gpu:
            model = model.cuda()

        n_chars = len(tokens.int_to_chars)

        losses = []

        for epoch in range(1, parameters.epochs + 1):
            runs = 0
            # initial hidden and cell state
            h, c = model.initial_hidden_state(parameters.batch_size)

            # train batch by batch
            for x, y in DataLoader.generate_batches(train_data,
                                                    parameters.batch_size,
                                                    parameters.window):

                runs += 1

                x = DataLoader.one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y).view(
                    parameters.batch_size * parameters.window)

                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    h, c = h.cuda(), c.cuda()

                # detach for BPTT :
                # If we don't, we'll back-prop all the way to the start
                h, c = h.detach(), c.detach()

                # zero out previous gradients
                model.zero_grad()

                # model output
                output, h, c = model(inputs, h, c)

                loss = criterion(output, targets)

                # back-propagation
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)
                optimizer.step()

                # model validation
                if runs % parameters.validation_counts == 0:
                    # run validation
                    hv, cv = model.initial_hidden_state(parameters.batch_size)

                    validation_losses = []

                    # enable evaluation mode
                    model.eval()

                    for val_x, val_y in DataLoader.generate_batches(
                            valid_data, parameters.batch_size,
                            parameters.window):
                        inputs = torch.from_numpy(
                            DataLoader.one_hot_encode(val_x, n_chars))
                        targets = torch.from_numpy(val_y).view(
                            parameters.batch_size * parameters.window)

                        if use_gpu:
                            inputs, targets = inputs.cuda(), targets.cuda()
                            hv, cv = hv.cuda(), cv.cuda()

                        hv, cv = hv.detach(), cv.detach()

                        output, hv, cv = model(inputs, hv, cv)

                        val_loss = criterion(output, targets)
                        validation_losses.append(val_loss.item())

                    train_loss = loss.item()
                    val_loss_final = np.mean(validation_losses)

                    logger.info(
                        f"Epoch: {epoch}/{runs} | Training loss: {train_loss}"
                        f" | Validation loss: {val_loss_final}")

                    losses.append({
                        "Epoch": epoch,
                        "Run": runs,
                        "TrainLoss": train_loss,
                        "ValidationLoss": val_loss_final
                    })

                    self._tb_writer.add_scalar("Loss/train", train_loss,
                                               epoch * 10000 + runs)
                    self._tb_writer.add_scalar("Loss/valid", val_loss_final,
                                               epoch * 10000 + runs)
                model.train()

            self._tb_writer.flush()
            self._save_check_point(model, parameters, tokens, epoch)

        self._save_check_point(model, parameters, tokens)

        return pd.DataFrame(losses)

    def _save_check_point(self, model: LSTMModel,
                          parameters: ModelHyperParameters,
                          tokens: Tokens, epoch: int = None):
        epoch_str = str(epoch) if epoch else "final"
        file_path, file_ext = os.path.splitext(self._save_path)
        checkpoint_file = f"{file_path}_{epoch_str}{file_ext}"
        logger.info(f"Saving checkpoint to file {checkpoint_file}")
        result = {
            "parameters": parameters.__dict__,
            "model": model.state_dict(),
            "tokens": tokens
        }
        torch.save(result, checkpoint_file)

    @staticmethod
    def _split_train_validation(data: np.array, validation_split: float):
        total_count = len(data)
        train_count, validation_count = int(
            total_count * (1 - validation_split)), int(
            total_count * validation_split)
        return data[:train_count], data[train_count:]
```

## Hyper-parameters

Below I define a python dataclass for all the hyper-parameters. I trained the model on a few different hyper-parameters until I settled on the below settings.

```python
@dataclass
class ModelHyperParameters:
    num_layers: int
    hidden_size: int
    epochs: int
    batch_size: int
    window: int
    learning_rate: float
    clip: float
    validation_split: float
    drop_prob: float
    validation_counts: int
    use_gpu: bool

parameters = {
  "num_layers": 2,
  "hidden_size": 512,
  "epochs": 30,
  "batch_size": 16,
  "window": 100,
  "learning_rate": 0.001,
  "clip": 5,
  "validation_split": 0.1,
  "drop_prob": 0.5,
  "validation_counts": 10,
  "use_gpu": True
}

parameters = ModelHyperParameters(**parameters)
```

Let's look at the training and validation results:

![Learning Curve](/assets/animal_farm_lstm_loss_values.png)

We can see from the validation loss function that the model has converged sufficiently.

## Sample Text
Now that we have trained the model, a great way to test it is to generate some sample text. We can initialize the model with a seed text and let the model generate new text based on the seed. For example, a good seed for Animal Farm could be "pigs", "animals", or "manor farm".


We can pass the output of the model through a softmax layer along the token dimension to check the activation for each character. We have a couple of options to generate new text:
- Use the topmost activated character as the next character
- Choose a random character among the top k activated characters

I went with Option 2. Let's look at some code:

### Predict Next Character

```python
def predict(model: LSTMModel, char: str, use_gpu: bool,
            h: torch.Tensor, c: torch.Tensor, top_k: int = 1):
    x = np.array([[model.tokens.chars_to_int[char]]])
    x = DataLoader.one_hot_encode(x, len(model.tokens.int_to_chars))
    inputs = torch.from_numpy(x)
    if use_gpu:
        inputs = inputs.cuda()
        model = model.cuda()

    h, c = h.detach(), c.detach()

    output, h, c = model(inputs, h, c)

    # Calculate softmax activation for each character
    p = functional.softmax(output, dim=1).data

    if use_gpu:
        p = p.cpu()

    # choose top k activate characters
    p, top_ch = p.topk(top_k)
    top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    # choose a random character based on their respective probabilities
    char_token = np.random.choice(top_ch, p=p / p.sum())

    return model.tokens.int_to_chars[char_token], h, c
```

We can now define a method that uses a seed and the predict function to generate new text:

```python
def generate_sample(model: LSTMModel, size: int, seed: str, top_k: int = 1,
                    use_gpu: bool = False) -> str:
    model.eval()  # eval mode

    text_chars = list(seed)
    h, c = model.initial_hidden_state(1)

    # go through the seed text to generate the next predicted character
    for i, char in enumerate(seed):
        next_char, h, c = predict(model=model, char=char,
                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)
        if i == len(seed) - 1:
            text_chars.append(next_char)

    # generate new text
    for i in range(size):
        next_char, h, c = predict(model=model, char=text_chars[-1],
                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)
        text_chars.append(next_char)

    return ''.join(text_chars)
```

## Results

Let's look at some sample text generated by the final model:

```
The two horses and his except out a dozen here were taken a arn end to a little distance, and the pigs as their seemed to and when the pigs were stall his men him and sever lear hat hands,
beside not what it was a straw and a sleap. His men had taken a stall. His mind
was some days. After her enough, he
said,
he was their own from any other to the fierce and the produce of the for the spelit of this windmill. To the stupy and the pigs almost the windmill had any again to the field to anyther on the farm which they had never been too that his starl. Sometimes to a proper of harms, set a knick which
who had been able to any frout to the five-barred gate, and the pigs, who went now an hour earlies and the disting of the farm and were some in a few minutes who
har head of the farm, and without the farm buildings, they could not to the fools, and with a commanding of anything to the windmill the animals had a great season of the with the
speak. His
steech were noting to any of the farm,
```

It sounds like gibberish, but, we can see that the model is able to put together different entities in the novel such as pigs, windmill, etc. The model has also learnt the structure of the novel and how it has broken text into paragraphs and some other aspects of the novel. For example, here's another text that was generated after epoch 7:

```
"Comrades!" he set the speech and the pigs were all the animals of their fives and the distance of the fire back and which had been set aside and winds of six and still his minds. I should harded that in a can the said with the farm buildings, and a sheep with this was a spot..
And we have
had a straight for himself, with his hoof, comrades.
```

Model identified some important patterns in the text such as "Comrades!" :). Another interesting one was:
George Orwell Bot

```
Chapter VI
All the animals were great some of the windmill. He had their eare and a stupid and the farm buildings, were all the animals. And there were setted and proppering a pilting of the farm, which was the oncasion were discurred the
windmill. And when the pigs and destail their
```

Model has learnt that the novel has chapters and they are in roman numerals.

## Bonus: 1984

As a follow up, I trained another lstm model on 1984's text. I changed a few hyper-parameters but the general structure looks the same. Let's look at some results from that model with the seed "The BIG BROTHER":

```
The BIG BROTHER I say the past of the street on the same was a comprested of the same of his been. There
was a side of a singed of his own straction. That was a sorn of to have to be a sorn of the same was to the street, the strunger and the same
was a sorn of the present of his matered and the production that had been a sorned of the starn of the street of the past of the stration of the past which was not the street on his man the
stall would be an any of the stratiction of the past was
all the past of the past of
```

## Github Link

You can access the full project on my [Github repo](https://github.com/kapilsh/ml-projects).
