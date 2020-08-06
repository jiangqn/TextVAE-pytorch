import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'sample'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

if args.task == 'train':
    from src.train.train_text_vae import train_vae
    train_vae(config)
elif args.task == 'test':
    from src.train.test_text_vae import test_vae
    test_vae(config)
elif args.task == 'sample':   # sample
    from src.sample_from_vae import sample_from_vae
    sample_from_vae(config)