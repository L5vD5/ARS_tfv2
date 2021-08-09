import argparse

from ARS import Agent

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--train', action="store_true", help='Train agent with given environment')
    parser.add_argument('--play', help="Play with a given weight directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    if args.train:
        a = Agent()
        print("Start training")
        a.train()
    elif args.play:
        a = Agent()
        print("Start playing")
        a.play(args.play)
