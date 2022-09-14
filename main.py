from Model import train

if __name__ == "__main__":
    train(max_steps=250, num_episodes=1000, model_path='models/old/game_156.pth')
