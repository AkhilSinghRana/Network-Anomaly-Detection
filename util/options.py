import argparse

def ArgumentParser():
        parser = argparse.ArgumentParser()

        parser.add_argument("--mode", type=str, default="train", help="train/test/continueTrain")
        
        #Arguments for reading data
        parser.add_argument("--fromCSV", type=bool, default=True, help="Read the data from CSV")
        parser.add_argument("--data_path", type=str, default="./data", help="Path to the CSV dataset")
        parser.add_argument("--visualize","-visualize", action="store_true", help="Enable data visualization module")

        #Arguments for RL algorithm/ taining / testing
        parser.add_argument("--model_name", type=str, default="autoencoder", help="Choose a model for training")
        parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train the learn function of algorithm")
        parser.add_argument("--exp_name", type=str, default="exp_1", help="experiment name (used to save model)")
        
        
        return args