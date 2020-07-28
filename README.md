# robot-nlp-spatial
Our goal is to build a NLP-guided Robot possessing the abilities to understand natural language instructions.

### Set up environment
1. Create a virtual environment with python version 3.6.5 using the following command:
``virtualenv venv -p <python_path>``

2. Activate the environment: ``source venv/bin/activate``
3. Install the required packages: ``pip install -r requirements.txt``
 

### How to run Lang-UNet model

``python train.py --batch_size <BATCH_SIZE> --epochs <NUM_EPOCHS> --lr <LEARNING_RATE``


### Baselines: How to run

1. Center : ``python center_baseline.py``
1. Random : ``python random_baseline.py``

#### Pipelined baselines
1. RNN-NoAttn-NoGround : ``python rnn_noattn.py``
1. LSTM-Attn-NoGround : ``python lstm_attn.py``
1. Lang-FCNet : ``python lang_fc.py``

#### End-to-end baselines
Download the extracted features from this [link](https://drive.google.com/drive/folders/15zwU9myZzb7dW7fx7kEvvgtDXZgdWrSg?usp=sharing) and place them inside the data/synthetic directory.

1. LSTM : ``python end_to_end.py --model lstm``
1. LSTM+CNN : ``python end_to_end.py --model cnn``
1. LSTM+CNN+SA : ``python end_to_end.py --model sa``
