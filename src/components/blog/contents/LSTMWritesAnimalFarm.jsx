import React from "react";
import {Collapse} from 'antd';
import {Typography} from "antd";
import {Card} from 'antd';
import {PythonSnippet} from "../snippets/PythonSnippet";
import {Comment, Avatar} from 'antd';
import {HTMLSnippet} from "../snippets/HTMLSnippet";
import lstmImage from "../../../../static/LSTM_WALL.png";
import lstmLossResultsImage from "../../../../static/animal_farm_lstm_loss_values.png";
import georgeOrwellImage from "../../../../static/george_orwell.jpg";
import {EditFilled} from "@ant-design/icons";
import mlProjectsSvg from "../../../../static/ml-projects.svg";


const {Title, Paragraph} = Typography;
const {Panel} = Collapse;
const {Meta} = Card;


const GeorgeOrwellBot = props =>
    <Comment
        author={"George Orwell Bot"}
        avatar={
            <Avatar
                src={georgeOrwellImage}
                alt="George Orwell Bot"
            />
        }
        content={
            <p style={{fontSize: 12}}>
                {props.text}
            </p>
        }
    />;

class LSTMWritesAnimalFarm extends React.Component {
    render() {
        return (
            <div>
                <Typography>
                    <Paragraph>
                        RNNs, and specially LSTMs are excellent models for language modelling. In this post, I will
                        train
                        an LSTM character by character to generate sample text from the famous novel <a
                        href={"https://en.wikipedia.org/wiki/Animal_Farm"}>Animal Farm</a> by <a
                        href={"https://en.wikipedia.org/wiki/George_Orwell"}>George Orwell</a>.
                    </Paragraph>
                    <Title level={3}>
                        Setup
                    </Title>
                    <Paragraph>
                        Before we start training, we need to load in the animal farm text and create a dataset that is
                        loadable into a PyTorch model. The full animal farm text is available <a
                        href={"https://github.com/kapilsh/ml-projects/blob/master/rnn/data/animal_farm.txt"}>here</a>.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def read_text_file(file_path: str) -> str:\n" +
                "    with open(file_path, 'r') as f:\n" +
                "        text = f.read()\n" +
                "    return text\n" +
                "file_path = ./data/animal_farm.txt\n" +
                "text = read_text_file(file_path)\n" +
                "print(text[:100])"}/>
                <HTMLSnippet
                    text={"'Chapter I\\n\\nMr. Jones, of the Manor Farm, had locked the hen-houses for the night, but\\nwas too drunk '"}/>
                <Typography>
                    <Paragraph>
                        Text in itself is hard to feed into a machine learning model, so typically it is <a
                        href={"https://en.wikipedia.org/wiki/One-hot"}>one hot encoded</a> into sparse vectors of [0, 1]
                        where numbers represent whether individual characters/words are
                        present.
                    </Paragraph>
                    <Paragraph>
                        Since we are training char by char, we will tokenize each character as a number. Below I write a
                        function to load all the text and
                        tokenize it:
                    </Paragraph>
                </Typography>
                <PythonSnippet
                    text={"Tokens = namedtuple(\"Tokens\", [\"int_to_chars\", \"chars_to_int\", \"tokens\"])\n" +
                    "\n" +
                    "def tokenize(text: str) -> Tokens:\n" +
                    "    chars = set(text)\n" +
                    "    int_to_chars = dict(enumerate(chars))\n" +
                    "    chars_to_int = {char: i for i, char in int_to_chars.items()}\n" +
                    "    tokens = np.array([chars_to_int[char] for char in text])\n" +
                    "    return Tokens(int_to_chars=int_to_chars, chars_to_int=chars_to_int,\n" +
                    "                  tokens=tokens)\n" +
                    "tokens = tokenize(text)\n" +
                    "print(tokens.tokens)\n"}/>
                <HTMLSnippet text={"array([26, 22, 15, 23, 24, 32, 27, 50, 33, 16, 16, 57, 27, 10, 50, 67, 39,\n" +
                "        7, 32, 47, 51, 50, 39, 68, 50, 24, 22, 32, 50, 57, 15,  7, 39, 27,\n" +
                "       50, 14, 15, 27, 64, 51, 50, 22, 15, 63, 50, 49, 39, 25, 29, 32, 63,\n" +
                "       50, 24, 22, 32, 50, 22, 32,  7, 12, 22, 39, 18, 47, 32, 47, 50, 68,\n" +
                "       39, 27, 50, 24, 22, 32, 50,  7, 56, 52, 22, 24, 51, 50, 42, 18, 24,\n" +
                "       16, 55, 15, 47, 50, 24, 39, 39, 50, 63, 27, 18,  7, 29, 50])"}/>
                <Typography>
                    <Title level={4}>
                        One Hot Encoding
                    </Title>
                    <Paragraph>
                        Now, let's write a method to one-hot encode our data. We will use this method later to encode
                        batches of data being fed into our RNN.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def one_hot_encode(tokens: np.array, label_counts: int) -> np.array:\n" +
                "    result = np.zeros((tokens.size, label_counts), dtype=np.float32)\n" +
                "    result[np.arange(result.shape[0]), tokens.flatten()] = 1\n" +
                "    result = result.reshape((*tokens.shape, label_counts))\n" +
                "    return result"}/>
                <Typography>
                    <Title level={4}>
                        Mini-batching
                    </Title>
                    <Paragraph>
                        Since we want to pass the data into our network in mini-batches, next step in the pre-processing
                        is to generate batches od data. In RNNs, we need to pass sequences as the mini-batches. Hence,
                        one way to batch is to split the full sequence into multiple sequences and then grab a window of
                        respective indices from all batches to feed into network.
                    </Paragraph>
                    <Paragraph>
                        For example, if the original sequence is of length 20, we can split that into 4 batches of
                        length 5 each. If our window size is 3, we can grab first 3 indices from the 4 batches to pass
                        into the network. Let's look at some code to do it.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def generate_batches(\n" +
                "        sequence: np.array, batch_size: int,\n" +
                "        window: int) -> Generator[Tuple[np.array, np.array], None, None]:\n" +
                "    batch_length = batch_size * window\n" +
                "    batch_count = len(sequence) // batch_length\n" +
                "\n" +
                "    truncated_size = batch_count * batch_length\n" +
                "    _sequence = sequence[:truncated_size]\n" +
                "    _sequence = _sequence.reshape((batch_size, -1))\n" +
                "\n" +
                "    for n in range(0, _sequence.shape[1], window):\n" +
                "        x = _sequence[:, n:n + window]\n" +
                "        y = np.zeros_like(x)\n" +
                "        if n < _sequence.shape[1]:\n" +
                "            y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, n + window]\n" +
                "        else:\n" +
                "            y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, 0]\n" +
                "        yield x, y"}/>
                <Typography>
                    <Paragraph>
                        Let's test the mini-batch implementation:
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"batches = generate_batches(tokens.tokens, 10, 40)\n" +
                "x, y = next(batches)\n" +
                "print(x[:5, :6])\n" +
                "print(y[:5, :5])"}/>
                <HTMLSnippet text={"array([[32, 61, 29, 28, 62, 10],\n" +
                "       [50, 24, 20, 34, 57, 51],\n" +
                "       [24, 29, 30, 57, 29, 24],\n" +
                "       [19, 45, 64,  5, 29, 20],\n" +
                "       [62, 61, 10, 57, 62, 61]])\n" +
                "\n" +
                "array([[61, 29, 28, 62, 10],\n" +
                "       [24, 20, 34, 57, 51],\n" +
                "       [29, 30, 57, 29, 24],\n" +
                "       [45, 64,  5, 29, 20],\n" +
                "       [61, 10, 57, 62, 61]])"}/>
                <Typography>
                    <Title level={3}>Long Short Term Memory (LSTM) Network</Title>
                    <Card
                        hoverable
                        cover={<img src={lstmImage} alt="woohoo"/>}
                        style={
                            {
                                width: "50%",
                                display: "block",
                                marginLeft: "auto",
                                marginRight: "auto",
                            }
                        }
                    >
                        <Meta title="LSTM Network Diagram" description={<span>Image Credits: <a
                            href={"https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714"}>
                            https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
                        </a></span>}/>
                    </Card>
                    <br/>
                    <Paragraph>
                        Next, we will define the LSTM model for our training. PyTorch provide a pre-built module for
                        LSTM so we can use that directly. After that we add a dropout layer for regularization followed
                        by a fully connected layer to receive model output. We also need to define what our
                        initial hidden and cell state will be. Let's implement the model class:
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"class LSTMModel(nn.Module):\n" +
                "    def __init__(self, tokens_size, **kwargs):\n" +
                "        super().__init__()\n" +
                "        self._drop_prob = kwargs.pop(\"drop_prob\")\n" +
                "        self._hidden_size = kwargs.pop(\"hidden_size\")\n" +
                "        self._num_layers = kwargs.pop(\"num_layers\")\n" +
                "\n" +
                "        self.lstm = nn.LSTM(\n" +
                "            input_size=tokens_size,\n" +
                "            hidden_size=self._hidden_size,\n" +
                "            num_layers=self._num_layers,\n" +
                "            dropout=self._drop_prob, batch_first=True)\n" +
                "\n" +
                "        self.dropout = nn.Dropout(self._drop_prob)\n" +
                "        self.fc = nn.Linear(self._hidden_size, tokens_size)\n" +
                "\n" +
                "    def forward(self, x, h, c):\n" +
                "        x_next, (hn, cn) = self.lstm(x, (h, c))\n" +
                "        x_dropout = self.dropout(x_next)\n" +
                "        x_stacked = x_dropout.contiguous().view(h.shape[1], -1,\n" +
                "                                                self._hidden_size)\n" +
                "        output = self.fc(x_stacked)\n" +
                "        return output, hn, cn\n" +
                "\n" +
                "    def initial_hidden_state(self, batch_size):\n" +
                "        # Initialize hidden state with zeros\n" +
                "        h0 = torch.zeros(self._num_layers, batch_size,\n" +
                "                         self._hidden_size).requires_grad_()\n" +
                "\n" +
                "        # Initialize cell state\n" +
                "        c0 = torch.zeros(self._num_layers, batch_size,\n" +
                "                         self._hidden_size).requires_grad_()\n" +
                "\n" +
                "        return h0, c0"}/>
                <Typography>
                    <Paragraph>
                        Now let's test the model by passing in a single batch of data.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"data_loader = DataLoader(\"rnn/data/animal_farm.txt\")\n" +
                "tokens = data_loader.tokens\n" +
                "batches = DataLoader.generate_batches(tokens.tokens, 10, 40)\n" +
                "x, y = next(batches)\n" +
                "x = DataLoader.one_hot_encode(x, n_chars)\n" +
                "y = DataLoader.one_hot_encode(y, n_chars)\n" +
                "inputs, targets = torch.from_numpy(x), torch.from_numpy(y)\n" +
                "print(inputs.shape)\n" +
                "print(targets.shape)\n" +
                "\n" +
                "model = LSTMModel(len(tokens.int_to_chars), drop_prob=0.1, num_layers=2, hidden_size=256)\n" +
                "h0, c0 = model.initial_hidden_state(batch_size)\n" +
                "output, hn, cn = model(inputs, h0, c0)\n" +
                "print(output.shape)"}/>
                <HTMLSnippet text={"torch.Size([10, 40, 71])\n" +
                "torch.Size([10, 40, 71])\n" +
                "torch.Size([10, 40, 71])"}/>
                <Typography>
                    <Paragraph>
                        Great! All the dimensions match. Now we can create a training routine to start training and
                        validating our model. I save the different checkpoints of the model after each epoch to compare
                        how the model improves after each epoch. During training, I also use SummaryWriter class from
                        PyTorch that allows
                        us to load results into <a href={"https://www.tensorflow.org/tensorboard"}>Tensorboard</a>
                    </Paragraph>
                </Typography>
                <Collapse accordian bordered={false} defaultActiveKey={['1']}
                          expandIcon={() => <EditFilled/>}>
                    <Panel header="Model Runner" key="1">
                        <PythonSnippet noBottomSpacing text={"class ModelRunner:\n" +
                        "    def __init__(self, data_loader: DataLoader, save_path: str):\n" +
                        "        self._data_loader = data_loader\n" +
                        "        self._save_path = save_path\n" +
                        "        self._tb_writer = SummaryWriter()\n" +
                        "\n" +
                        "    def train(self, parameters: ModelHyperParameters):\n" +
                        "        use_gpu = parameters.use_gpu and torch.cuda.is_available()\n" +
                        "        if use_gpu:\n" +
                        "            logger.info(\"GPU Available and Enabled: Using CUDA\")\n" +
                        "        else:\n" +
                        "            logger.info(\"GPU Disabled: Using CPU\")\n" +
                        "\n" +
                        "        # load the tokens from the text\n" +
                        "        tokens = self._data_loader.tokens\n" +
                        "\n" +
                        "        # define the model\n" +
                        "        model = LSTMModel(tokens=tokens,\n" +
                        "                          drop_prob=parameters.drop_prob,\n" +
                        "                          num_layers=parameters.num_layers,\n" +
                        "                          hidden_size=parameters.hidden_size)\n" +
                        "\n" +
                        "        # enable training mode\n" +
                        "        model.train()\n" +
                        "\n" +
                        "        # use Adam optimizer\n" +
                        "        optimizer = torch.optim.Adam(model.parameters(),\n" +
                        "                                     lr=parameters.learning_rate)\n" +
                        "\n" +
                        "        # loss function\n" +
                        "        criterion = nn.CrossEntropyLoss()\n" +
                        "\n" +
                        "        # split data into training and validation sets\n" +
                        "        train_data, valid_data = self._split_train_validation(\n" +
                        "            tokens.tokens, parameters.validation_split)\n" +
                        "\n" +
                        "        if use_gpu:\n" +
                        "            model = model.cuda()\n" +
                        "\n" +
                        "        n_chars = len(tokens.int_to_chars)\n" +
                        "\n" +
                        "        losses = []\n" +
                        "\n" +
                        "        for epoch in range(1, parameters.epochs + 1):\n" +
                        "            runs = 0\n" +
                        "            # initial hidden and cell state\n" +
                        "            h, c = model.initial_hidden_state(parameters.batch_size)\n" +
                        "\n" +
                        "            # train batch by batch\n" +
                        "            for x, y in DataLoader.generate_batches(train_data,\n" +
                        "                                                    parameters.batch_size,\n" +
                        "                                                    parameters.window):\n" +
                        "\n" +
                        "                runs += 1\n" +
                        "\n" +
                        "                x = DataLoader.one_hot_encode(x, n_chars)\n" +
                        "                inputs, targets = torch.from_numpy(x), torch.from_numpy(y).view(\n" +
                        "                    parameters.batch_size * parameters.window)\n" +
                        "\n" +
                        "                if use_gpu:\n" +
                        "                    inputs, targets = inputs.cuda(), targets.cuda()\n" +
                        "                    h, c = h.cuda(), c.cuda()\n" +
                        "\n" +
                        "                # detach for BPTT :\n" +
                        "                # If we don't, we'll back-prop all the way to the start\n" +
                        "                h, c = h.detach(), c.detach()\n" +
                        "\n" +
                        "                # zero out previous gradients\n" +
                        "                model.zero_grad()\n" +
                        "\n" +
                        "                # model output\n" +
                        "                output, h, c = model(inputs, h, c)\n" +
                        "\n" +
                        "                loss = criterion(output, targets)\n" +
                        "\n" +
                        "                # back-propagation\n" +
                        "                loss.backward()\n" +
                        "\n" +
                        "                # gradient clipping\n" +
                        "                nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)\n" +
                        "                optimizer.step()\n" +
                        "\n" +
                        "                # model validation\n" +
                        "                if runs % parameters.validation_counts == 0:\n" +
                        "                    # run validation\n" +
                        "                    hv, cv = model.initial_hidden_state(parameters.batch_size)\n" +
                        "\n" +
                        "                    validation_losses = []\n" +
                        "\n" +
                        "                    # enable evaluation mode\n" +
                        "                    model.eval()\n" +
                        "\n" +
                        "                    for val_x, val_y in DataLoader.generate_batches(\n" +
                        "                            valid_data, parameters.batch_size,\n" +
                        "                            parameters.window):\n" +
                        "                        inputs = torch.from_numpy(\n" +
                        "                            DataLoader.one_hot_encode(val_x, n_chars))\n" +
                        "                        targets = torch.from_numpy(val_y).view(\n" +
                        "                            parameters.batch_size * parameters.window)\n" +
                        "\n" +
                        "                        if use_gpu:\n" +
                        "                            inputs, targets = inputs.cuda(), targets.cuda()\n" +
                        "                            hv, cv = hv.cuda(), cv.cuda()\n" +
                        "\n" +
                        "                        hv, cv = hv.detach(), cv.detach()\n" +
                        "\n" +
                        "                        output, hv, cv = model(inputs, hv, cv)\n" +
                        "\n" +
                        "                        val_loss = criterion(output, targets)\n" +
                        "                        validation_losses.append(val_loss.item())\n" +
                        "\n" +
                        "                    train_loss = loss.item()\n" +
                        "                    val_loss_final = np.mean(validation_losses)\n" +
                        "\n" +
                        "                    logger.info(\n" +
                        "                        f\"Epoch: {epoch}/{runs} | Training loss: {train_loss}\"\n" +
                        "                        f\" | Validation loss: {val_loss_final}\")\n" +
                        "\n" +
                        "                    losses.append({\n" +
                        "                        \"Epoch\": epoch,\n" +
                        "                        \"Run\": runs,\n" +
                        "                        \"TrainLoss\": train_loss,\n" +
                        "                        \"ValidationLoss\": val_loss_final\n" +
                        "                    })\n" +
                        "\n" +
                        "                    self._tb_writer.add_scalar(\"Loss/train\", train_loss,\n" +
                        "                                               epoch * 10000 + runs)\n" +
                        "                    self._tb_writer.add_scalar(\"Loss/valid\", val_loss_final,\n" +
                        "                                               epoch * 10000 + runs)\n" +
                        "                model.train()\n" +
                        "\n" +
                        "            self._tb_writer.flush()\n" +
                        "            self._save_check_point(model, parameters, tokens, epoch)\n" +
                        "\n" +
                        "        self._save_check_point(model, parameters, tokens)\n" +
                        "\n" +
                        "        return pd.DataFrame(losses)\n" +
                        "\n" +
                        "    def _save_check_point(self, model: LSTMModel,\n" +
                        "                          parameters: ModelHyperParameters,\n" +
                        "                          tokens: Tokens, epoch: int = None):\n" +
                        "        epoch_str = str(epoch) if epoch else \"final\"\n" +
                        "        file_path, file_ext = os.path.splitext(self._save_path)\n" +
                        "        checkpoint_file = f\"{file_path}_{epoch_str}{file_ext}\"\n" +
                        "        logger.info(f\"Saving checkpoint to file {checkpoint_file}\")\n" +
                        "        result = {\n" +
                        "            \"parameters\": parameters.__dict__,\n" +
                        "            \"model\": model.state_dict(),\n" +
                        "            \"tokens\": tokens\n" +
                        "        }\n" +
                        "        torch.save(result, checkpoint_file)\n" +
                        "\n" +
                        "    @staticmethod\n" +
                        "    def _split_train_validation(data: np.array, validation_split: float):\n" +
                        "        total_count = len(data)\n" +
                        "        train_count, validation_count = int(\n" +
                        "            total_count * (1 - validation_split)), int(\n" +
                        "            total_count * validation_split)\n" +
                        "        return data[:train_count], data[train_count:]"}/>
                    </Panel>
                </Collapse>
                <br/>
                <Typography>
                    <Title level={4}>
                        Hyper-parameters
                    </Title>
                    <Paragraph>
                        Below I define a python dataclass for all the hyper-parameters. I trained the model on a few
                        different hyper-parameters until I settled on the below settings.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"@dataclass\n" +
                "class ModelHyperParameters:\n" +
                "    num_layers: int\n" +
                "    hidden_size: int\n" +
                "    epochs: int\n" +
                "    batch_size: int\n" +
                "    window: int\n" +
                "    learning_rate: float\n" +
                "    clip: float\n" +
                "    validation_split: float\n" +
                "    drop_prob: float\n" +
                "    validation_counts: int\n" +
                "    use_gpu: bool\n" +
                "\n" +
                "parameters = {\n" +
                "  \"num_layers\": 2,\n" +
                "  \"hidden_size\": 512,\n" +
                "  \"epochs\": 30,\n" +
                "  \"batch_size\": 16,\n" +
                "  \"window\": 100,\n" +
                "  \"learning_rate\": 0.001,\n" +
                "  \"clip\": 5,\n" +
                "  \"validation_split\": 0.1,\n" +
                "  \"drop_prob\": 0.5,\n" +
                "  \"validation_counts\": 10,\n" +
                "  \"use_gpu\": True\n" +
                "}\n" +
                "\n" +
                "parameters = ModelHyperParameters(**parameters)"}/>
                <Typography>
                    <Paragraph>
                        Let's look at the training and validation results:
                    </Paragraph>
                </Typography>
                <img
                    alt="Loss Values"
                    src={lstmLossResultsImage}
                    style={{
                        width: "80%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Typography>
                    <Paragraph>
                        We can see from the validation loss function that the model has converged sufficiently.
                    </Paragraph>
                    <Title level={3}>
                        Sample Text
                    </Title>
                    <Paragraph>
                        Now that we have trained the model, a great way to test it is to generate some sample text. We
                        can initialize the model with a seed text and let the model generate new text based on the seed.
                        For example, a good seed for Animal Farm could be "pigs", "animals", or "manor farm".
                    </Paragraph>
                    <Paragraph>
                        We can pass the output of the model through a softmax layer along the token dimension to check
                        the activation for each character. We have a couple of options to generate new text:
                        <ul>
                            <li>
                                Use the topmost activated character as the next character
                            </li>
                            <li>
                                Choose a random character among the top k activated characters
                            </li>
                        </ul>
                        I went with Option 2. Let's look at some code:
                    </Paragraph>
                </Typography>
                <Collapse accordian bordered={false} defaultActiveKey={['1']}
                          expandIcon={() => <EditFilled/>}>
                    <Panel header="Predict Next Character" key="1">
                        <PythonSnippet noBottomSpacing
                                       text={"def predict(model: LSTMModel, char: str, use_gpu: bool,\n" +
                                       "            h: torch.Tensor, c: torch.Tensor, top_k: int = 1):\n" +
                                       "    x = np.array([[model.tokens.chars_to_int[char]]])\n" +
                                       "    x = DataLoader.one_hot_encode(x, len(model.tokens.int_to_chars))\n" +
                                       "    inputs = torch.from_numpy(x)\n" +
                                       "    if use_gpu:\n" +
                                       "        inputs = inputs.cuda()\n" +
                                       "        model = model.cuda()\n" +
                                       "\n" +
                                       "    h, c = h.detach(), c.detach()\n" +
                                       "\n" +
                                       "    output, h, c = model(inputs, h, c)\n" +
                                       "\n" +
                                       "    # Calculate softmax activation for each character\n" +
                                       "    p = functional.softmax(output, dim=1).data\n" +
                                       "\n" +
                                       "    if use_gpu:\n" +
                                       "        p = p.cpu()\n" +
                                       "\n" +
                                       "    # choose top k activate characters\n" +
                                       "    p, top_ch = p.topk(top_k)\n" +
                                       "    top_ch = top_ch.numpy().squeeze()\n" +
                                       "\n" +
                                       "    p = p.numpy().squeeze()\n" +
                                       "    # choose a random character based on their respective probabilities\n" +
                                       "    char_token = np.random.choice(top_ch, p=p / p.sum())\n" +
                                       "\n" +
                                       "    return model.tokens.int_to_chars[char_token], h, c"}/>
                    </Panel>
                </Collapse>
                <br/>
                <Typography>
                    <Paragraph>
                        We can now define a method that uses a seed and the predict function to generate new text:
                    </Paragraph>
                </Typography>
                <Collapse accordian bordered={false} defaultActiveKey={['1']}
                          expandIcon={() => <EditFilled/>}>
                    <Panel header="Generate Sample Text" key="1">
                        <PythonSnippet noBottomSpacing
                                       text={"def generate_sample(model: LSTMModel, size: int, seed: str, top_k: int = 1,\n" +
                                       "                    use_gpu: bool = False) -> str:\n" +
                                       "    model.eval()  # eval mode\n" +
                                       "\n" +
                                       "    text_chars = list(seed)\n" +
                                       "    h, c = model.initial_hidden_state(1)\n" +
                                       "\n" +
                                       "    # go through the seed text to generate the next predicted character\n" +
                                       "    for i, char in enumerate(seed):\n" +
                                       "        next_char, h, c = predict(model=model, char=char,\n" +
                                       "                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)\n" +
                                       "        if i == len(seed) - 1:\n" +
                                       "            text_chars.append(next_char)\n" +
                                       "\n" +
                                       "    # generate new text\n" +
                                       "    for i in range(size):\n" +
                                       "        next_char, h, c = predict(model=model, char=text_chars[-1],\n" +
                                       "                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)\n" +
                                       "        text_chars.append(next_char)\n" +
                                       "\n" +
                                       "    return ''.join(text_chars)"}/>
                    </Panel>
                </Collapse>
                <br/>
                <Typography>
                    <Title level={3}>
                        Results
                    </Title>
                    <Paragraph>
                        Let's look at some sample text generated by the final model:
                    </Paragraph>
                    <GeorgeOrwellBot
                        text={"The two horses and his except out a dozen here were taken a arn end to a little distance, and the pigs as their seemed to and when the pigs were stall his men him and sever lear hat hands,\n" +
                        "beside not what it was a straw and a sleap. His men had taken a stall. His mind\n" +
                        "was some days. After her enough, he\n" +
                        "said,\n" +
                        "he was their own from any other to the fierce and the produce of the for the spelit of this windmill. To the stupy and the pigs almost the windmill had any again to the field to anyther on the farm which they had never been too that his starl. Sometimes to a proper of harms, set a knick which\n" +
                        "who had been able to any frout to the five-barred gate, and the pigs, who went now an hour earlies and the disting of the farm and were some in a few minutes who\n" +
                        "har head of the farm, and without the farm buildings, they could not to the fools, and with a commanding of anything to the windmill the animals had a great season of the with the\n" +
                        "speak. His\n" +
                        "steech were noting to any of the farm,"}/>
                    <Paragraph>
                        It sounds like gibberish, but, we can see that the model is able to put together different
                        entities in the novel such as pigs, windmill, etc. The model has also learnt the
                        structure of the novel and how it has broken text into paragraphs and some other
                        aspects of the novel. For example, here's another text that was generated after epoch 7:
                    </Paragraph>
                    <GeorgeOrwellBot
                        text={"\"Comrades!\" he set the speech and the pigs were all the animals of their fives and the distance of the fire back and which had been set aside and winds of six and still his minds. I should harded that in a can the said with the farm buildings, and a sheep with this was a spot..\n" +
                        "And we have\n" +
                        "had a straight for himself, with his hoof, comrades.\n"}/>
                    <Paragraph>
                        Model identified some important patterns in the text such as "Comrades!" :). Another interesting
                        one was:
                    </Paragraph>
                    <GeorgeOrwellBot text={"Chapter VI\n" +
                    "All the animals were great some of the windmill. He had their eare and a stupid and the farm buildings, were all the animals. And there were setted and proppering a pilting of the farm, which was the oncasion were discurred the\n" +
                    "windmill. And when the pigs and destail their"}/>
                    <Paragraph>
                        Model has learnt that the novel has chapters and they are in roman numerals.
                    </Paragraph>
                    <Title level={3}>
                        Bonus: 1984
                    </Title>
                    <Paragraph>
                        As a follow up, I trained another lstm model on 1984's text. I changed a few hyper-parameters
                        but
                        the general structure looks the same. Let's look at some results from that model with the seed
                        "The BIG BROTHER":
                    </Paragraph>
                    <GeorgeOrwellBot
                        text={"The BIG BROTHER I say the past of the street on the same was a comprested of the same of his been. There\n" +
                        "was a side of a singed of his own straction. That was a sorn of to have to be a sorn of the same was to the street, the strunger and the same\n" +
                        "was a sorn of the present of his matered and the production that had been a sorned of the starn of the street of the past of the stration of the past which was not the street on his man the\n" +
                        "stall would be an any of the stratiction of the past was\n" +
                        "all the past of the past of"}/>
                    <Title level={3}>
                        Github Link
                    </Title>
                    <Paragraph>
                        You can access the full project on my Github repo in the rnn directory:
                    </Paragraph>
                    <a href={"https://github.com/kapilsh/ml-projects"}>
                        <img
                            alt="ML Projects Repo"
                            src={mlProjectsSvg}
                            style={{
                                width: "35%",
                                display: "block",
                                marginLeft: "auto",
                                marginRight: "auto",
                            }}
                        />
                    </a>
                </Typography>
            </div>
        );
    }
}

export default LSTMWritesAnimalFarm;