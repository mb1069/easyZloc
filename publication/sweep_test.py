
import wandb

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--var1')
    parser.add_argument('--var2')
    args = vars(parser.parse_args())

    wandb.init(
        project='test',
        config = {
            'var1': args['var1'],
            'var2': args['var2']
        },
    )
    wandb.log({'var3': args['var1'] + args['var2']})
