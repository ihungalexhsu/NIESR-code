from seq2seq import Seq2seq
from uai_seq2seq import UAI_seq2seq
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-model', '-m', 
                        choices=['uai', 'seq2seq'],
                        default='uai')
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--beta', dest='beta', action='store', type=str)
    parser.add_argument('--gamma', dest='gamma', action='store', type=str)
    parser.add_argument('--delta', dest='delta', action='store', type=str)
    parser.add_argument('--z', dest='z', action='store_true')
    args = parser.parse_args()
    # alpha, set in config file, represents the value to multiply on main prediction
    # beta represents the value to multiply on reconstruction task
    # gamma represents the value to multiply on disentenglement task
    # delta represents the dropout rate on the "add-noise" layer
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    state_dict = None
    if args.load_model:
        if args.model =='uai':
            model = UAI_seq2seq(config, args.beta, args.gamma, args.delta, load_model=True)
        elif args.model =='seq2seq':
            model = Seq2seq(config, load_model=True)
    else:
        if args.model =='uai':
            model = UAI_seq2seq(config, args.beta, args.gamma, args.delta, load_model=False)
        elif args.model =='seq2seq':
            model = Seq2seq(config, load_model=False)
    
    if args.test:
        if args.train:
            state_dict, cer = model.train()
            model.test(state_dict)

        else:
            model.test()
    if args.z:
        model.get_z_accuracy_data(state_dict)
