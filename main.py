import os
import shutil
import sys
import yaml
import argparse
import logging
from src.data_csv_to_client import make_data, plot_data_distribution

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", type=str, required=True)
    parser.add_argument("--output", help="output folder name", type=str, required=False)
    parser.add_argument("-f", "--force", help="force overwrite of existing output folder", action="store_true")
    parser.add_argument("-r", "--redo_data", help="force redo of data extraction", action="store_true")
    args = parser.parse_args()  


    logging.info("Loading config file...")
    try:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            evaluation_type = config["evaluation_type"]
            num_clients = config["num_clients"]
            num_rounds = config["num_rounds"]
            data_name = config["data_name"]
            dataset_folder = config["dataset_folder"]
            train_filename = config["train_filename"]
            test_filename = config["test_filename"]
            clients_special_distribution = config["clients_special_distribution"]
            seed = config["seed"]
            batch_size = config["batch_size"]
            num_epochs = config["num_epochs"]
            validation_split = config["validation_split"]
    except:
        logging.error("Error loading config file !")
        logging.error(sys.exc_info()[0])
        logging.error("Exiting...")
        exit()
    logging.info("Config file loaded !")

    # Prepare the data for each client
    logging.info("Extracting data for each client...")
    data_folder = "data_client_" + data_name
    if not os.path.exists(data_folder) or args.redo_data:
        if os.path.exists(data_folder):
            logging.info("Overwriting existing data folder...")
            shutil.rmtree(data_folder)
        else:
            logging.info("No data folder named {} found ! Creating it...".format(data_folder))
        num_clients = num_clients
        train_filepath = os.path.join("dataset", dataset_folder, train_filename)
        test_filepath = os.path.join("dataset", dataset_folder, test_filename)
        try:
            make_data(num_clients, train_filepath, test_filepath, data_folder, clients_special_distribution, seed=seed)
        except FileNotFoundError:
            logging.error("Error loading data file ! Please check the train_filename and test_filename in the config file.")
            os.rmdir(data_folder)
            logging.error("Exiting...")
            exit()
        plot_data_distribution(num_clients, data_folder)
        logging.info("Data extracted !")
    else:
        if len(os.listdir(data_folder)) != num_clients + 1:
            logging.error("Data folder {} already exists but does not contain the correct number of files !".format(data_folder))
            logging.error("This can be due to the unmatched number of clients between the config file and the data folder.")
            logging.error("Please delete it and run this script again.")
            logging.error("Exiting...")
            exit() 
            
        logging.info("Using existing data folder called {}.".format(data_folder))


    # Create the output folder
    if not args.output:
        args.output = "final_" + evaluation_type + "_" + args.config.split(".")[0]
    else:
        args.output = "final_" + evaluation_type + "_" + args.output

    if os.path.exists(args.output):
        logging.warning("Output folder already exists !")
        if args.force:
            logging.warning("Overwriting existing folder (-f flag)")
            shutil.rmtree(args.output)
            os.makedirs(args.output)
        else:
            resp = input("Do you want to overwrite it ? (y/n)")
            if resp == "y":
                logging.info("Erasing existing folder...")
                shutil.rmtree(args.output)
                os.makedirs(args.output)
            else:
                logging.info("Exiting...")
                exit()
    else:
        os.makedirs(args.output)

    # Start the simulation
    logging.info("Starting simulation...")
    if evaluation_type == "centralized":
        logging.info("Running centralized simulation...")
        os.system("python src/main_ce.py -o {} -d {} --exp --batch_size {} --num_epochs {} --num_rounds {} --num_clients {} --validation_split {}".format('_'.join(args.output.split("_")[2:]), data_name, batch_size, num_epochs, num_rounds, num_clients, validation_split))
    elif evaluation_type == "decentralized":
        logging.info("Running decentralized simulation...")
        # TODO
    else:
        logging.error("Invalid evaluation type !")
        logging.error("Exiting...")
        exit()
    logging.info("Simulation finished !")
    logging.info("Results saved in {}.".format(args.output))
    

