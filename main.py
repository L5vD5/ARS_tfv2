import argparse

from ARS import Agent

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--train', action="store_true", help='Train agent with given environment')
    parser.add_argument('--play', help="Play with a given weight directory")
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of worker in ray')
    parser.add_argument('--total_steps', default=1000, help='Number of epochs of interaction (equivalent to number of policy updates) to perform')
    parser.add_argument('--hdims', type=int, nargs='+', help='size of hidden dimension', required=True)
    parser.add_argument('--max_ep_len_eval', default=1e3, help='Maximum length of trajectory')
    parser.add_argument('--num_eval', default=3, help='How many steps per epoch')
    parser.add_argument('--print_every', default=10, help='How often to print result')
    parser.add_argument('--ep_len_rollout', default=1000, help='How many episodes per rollout in ray')
    parser.add_argument('--actv', default='tanh', help='Activation function')
    parser.add_argument('--out_actv', default='tanh', help='Activation function in output layer')
    parser.add_argument('--evaluate_every', default=50,  help='How often to evaluate agent')
    parser.add_argument('--alpha', default=0.01,  help='Fixed step size')
    parser.add_argument('--nu', default=0.06,  help='Standard deviation of noise')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    if args.train:
        a = Agent(args)
        print("Start training")
        a.train()
    elif args.play:
        a = Agent(args)
        print("Start playing")
        a.play(args.play)
