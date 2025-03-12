import pandas as pd
import numpy as np
import os


class DataCleaner:
    """
    Class for Data Cleaning Functions, using it as initial preprocessing of files
    """

    @staticmethod
    def preProcessFiles(data_path: str, clean_path: str):
        """
        Loops through the folder, searching for relevant files and making sure the columns are as described in the
        readme
        :param data_path:
        :param clean_path:
        :return:
        """

        operational_names = ['settings 1', 'settings 2', 'settings 3']
        sensor_names = ['sensor {}'.format(i) for i in range(1, 22)]

        coloumns = ['unit number', 'time cycles'] + operational_names + sensor_names



        for file in os.listdir(data_path):
            if file.endswith('.txt') and not (file.startswith('RUL') or file.startswith('readme')):
                df = pd.read_csv(data_path + file, sep='\s+', header=None, names=coloumns)
                file = file.replace('.txt', '.csv')
                df.to_csv(clean_path + file, index=False)

def main():

    data_path = 'CMAPSSData/'
    clean_path = 'clean_data/'

    cleaner = DataCleaner()
    cleaner.preProcessFiles(data_path, clean_path)



if __name__ == '__main__':
    main()