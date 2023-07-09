# Re-Creation of Gov2Vec Model

This is an attempt to re-create the a paper by John J Nay titled [Gov2Vec: Learning Distributed Representations of Institutions and Their Legal Text](https://arxiv.org/abs/1609.06616).

The model uses a Continuous-Bag-of-Words, along with the government institution that produces those words, to predict a target words. During the process of training, both word embeddings and institution embeddings are learned as a predictive features for guessing a target word. This Distributed Representation Model is presented by [Le and Mikolov (2014)](https://arxiv.org/abs/1405.4053) and expanded by [Dai et al. (2015)](https://arxiv.org/abs/1507.07998).

The embeddings of the trained model can then be used to interpret queries using vector arithmetic.

```bash
python main.py --mps --epochs 6             # Train an AVG-AVG model with MPS
python main.py --mps --epochs 6 --arch-gov CONCAT
                                            # Train an AVG-CONCAT model with MPS
```

## Usage

During training, if a keyboard iterrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the testing queries.

The `main.py` script accepts the following:

```bash
options:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --window WINDOW       context window length from both sizes of the target word
  --arch-context CONTEXT
                        architecture of context combination (AVG, SUM, CONCAT)
  --arch-gov GOV        architecture of government combination (AVG, SUM, CONCAT)
  --emsize EMSIZE       size of word and gov embeddings
  --lr LR               initial learning rate
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --seed SEED           random seed
  --cuda                use CUDA
  --mps                 enables macOS GPU training
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --dry-run             verify the code and the model
```