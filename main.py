from net import network
import argparse

def run(pattern):
    net = network(pattern)
    net.build_net()
    net.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern', type=str, default='test',  # pattern model:test or train
                        required=True, help='Choice train or test model')

    args = parser.parse_args()
    print("Run One Shot Learning Model with Omniglot Datasets for " + args.pattern)
    run(args.pattern)

# RUN:
# python main.py --pattern train
# python main.py --pattern test


# -------------------------------------------------------------END-------------------------------------------------------------
