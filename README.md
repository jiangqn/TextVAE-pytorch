# TextVAE-pytorch
Implementation of Variational Auto-Encoder for text generation in pytorch.

**Reference**: Generating Sentences from a Continuous Space. ACL 2016. [[paper]](https://arxiv.org/pdf/1511.06349.pdf)

## Usage

Modify model and sample configuration in ```config.yaml``` or write new configuration file.
Specify task (train, test, sample), gpu and config in command.

### Train

```
python main.py --task train
```

### Test

```
python main.py --task test
```

### Sample

```
python main.py --task sample
```
