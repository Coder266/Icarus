from Model import train_sl

if __name__ == "__main__":
    train_sl(['/home/andre/Documents/MyBot/datasets/standard_no_press.jsonl',
              '/home/andre/Documents/MyBot/datasets/standard_press_without_msgs.jsonl'])
             # model_path='/home/andre/Documents/MyBot/models/sl_model_DipNet_1_336.pth')
