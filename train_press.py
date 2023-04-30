from training.SL_msg_train import train_msg_sl
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='MyBot - Diplomacy AI')
    parser.add_argument('dataset_path', type=str,
                        help='jsonlines file with games to learn from')
    parser.add_argument('-m', '--model_path', help='pre-trained model to initialize model before training (.pth file)',
                        type=str, default=None)
    parser.add_argument('-g', '--gunboat_model_path', type=str, default=None,
                        help='pre-trained gunboat model to initialize the encoder and the value network (.pth file)')
    parser.add_argument('-p', '--print_ratio', help='how often to print loss and accuracy, 0 disables printing',
                        type=int, default=0)
    parser.add_argument('-s', '--save_ratio', help='how often to save the model, 0 disables saving',
                        type=int, default=0)
    parser.add_argument('-o', '--output_header', help='header for the name of the output model file, default is sl_',
                        type=str, default='sl')
    parser.add_argument('-l', '--log_file', help='name of the log file',
                        type=str, default=None)
    parser.add_argument('-d', '--dist_lr',
                        help='learning rate for the policy distribution layers of the model, default is 1e-4',
                        type=float, default=1e-5)
    parser.add_argument('-e', '--embed_size',
                        help='size of the embedding for the board state and previous orders, default is 224',
                        type=int, default=224)
    parser.add_argument('--msg_embed_size',
                        help='size of the embedding for each press message, default is 100',
                        type=int, default=100)
    parser.add_argument('-t', '--transformer_layers', help='number of transformer layers, default is 5',
                        type=int, default=5)
    parser.add_argument('--validation_size', help='number of games to keep in the validation set, default is 20',
                        type=int, default=20)
    parser.add_argument('--transformer_heads', help='number of transformer multi-attention heads, default is 8',
                        type=int, default=8)
    parser.add_argument('--lstm_size', help='width of lstm, default is 200',
                        type=int, default=200)
    parser.add_argument('--lstm_layers', help='number of lstm layers, default is 2',
                        type=int, default=2)
    parser.add_argument('--msg_log_size', help='how many past messages to use for making decisions, default is 20',
                        type=int, default=20)
    parser.add_argument('--restore_game', help='restore checkpoint and start from this game',
                        type=int, default=None)
    parser.add_argument('--restore_epoch', help='restore checkpoint and start from this epoch',
                        type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_msg_sl(dataset_path=args.dataset_path, model_path=args.model_path, gunboat_model_path=args.gunboat_model_path,
                 print_ratio=args.print_ratio, save_ratio=args.save_ratio, output_header=args.output_header,
                 log_file=args.log_file,
                 dist_learning_rate=args.dist_lr, validation_size=args.validation_size,
                 embed_size=args.embed_size, msg_embed_size=args.msg_embed_size,
                 transformer_layers=args.transformer_layers, transformer_heads=args.transformer_heads,
                 lstm_size=args.lstm_size, lstm_layers=args.lstm_layers, msg_log_size=args.msg_log_size,
                 restore_game=args.restore_game, restore_epoch=args.restore_epoch)
