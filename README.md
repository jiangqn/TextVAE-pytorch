# TextVAE-pytorch
Implementation of Variational Auto-Encoder for text generation in pytorch.

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