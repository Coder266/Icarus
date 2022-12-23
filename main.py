from SL_train import train_sl
import argparse
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description='MyBot - Diplomacy AI')
    parser.add_argument('dataset_path', type=str,
                        help='jsonlines file with games to learn from')
    parser.add_argument('-g', '--gpu', help='gpu id for cuda to use',
                        type=int, default=0)
    parser.add_argument('-m', '--model_path', help='pre-trained model to initialize model before training (.pth file)',
                        type=str, default=None)
    parser.add_argument('-p', '--print_ratio', help='how often to print loss and accuracy, 0 disables printing',
                        type=int, default=0)
    parser.add_argument('-s', '--save_ratio', help='how often to save the model, 0 disables saving',
                        type=int, default=0)
    parser.add_argument('-o', '--output_header', help='header for the name of the output model file, default is sl_',
                        type=str, default='sl_')
    parser.add_argument('-l', '--log_file', help='name of the log file',
                        type=str, default=None)
    parser.add_argument('-d', '--dist_lr',
                        help='learning rate for the policy distribution layers of the model, default is 1e-4',
                        type=float, default=1e-5)
    parser.add_argument('-v', '--value_lr', help='learning rate for the value layers of the model, default is 1e-6',
                        type=float, default=1e-6)
    parser.add_argument('-e', '--embed_size',
                        help='size of the embedding for the board state and previous orders, default is 224',
                        type=int, default=224)
    parser.add_argument('-t', '--transformer_layers', help='number of transformer layers, default is 10',
                        type=int, default=10)
    parser.add_argument('--validation_size', help='number of games to keep in the validation set, default is 200',
                        type=int, default=200)
    parser.add_argument('--transformer_heads', help='number of transformer multi-attention heads, default is 8',
                        type=int, default=8)
    parser.add_argument('--lstm_size', help='width of lstm, default is 200',
                        type=int, default=200)
    parser.add_argument('--lstm_layers', help='number of lstm layers, default is 2',
                        type=int, default=2)
    parser.add_argument('--restore_game', help='restore checkpoint and start from this game',
                        type=int, default=None)
    parser.add_argument('--restore_epoch', help='restore checkpoint and start from this epoch',
                        type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_sl(dataset_path=args.dataset_path,
             model_path=args.model_path, print_ratio=args.print_ratio, save_ratio=args.save_ratio,
             output_header=args.output_header, log_file=args.log_file,
             dist_learning_rate=args.dist_lr, value_learning_rate=args.value_lr, validation_size=args.validation_size,
             embed_size=args.embed_size, transformer_layers=args.transformer_layers,
             transformer_heads=args.transformer_heads, lstm_size=args.lstm_size, lstm_layers=args.lstm_layers,
             restore_game=args.restore_game, restore_epoch=args.restore_epoch)
