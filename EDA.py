import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalysis:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def plotSettingsToTime(self):
        """
        General function to plot each of the settings against the time cycle to see how much it deviates in order to
        determine if this is a variable impacting degradation
        :return: Plots of the settings
        """
        file_path = self.folder_path + 'train_FD001.csv'
        print('Reading the file')

        df = pd.read_csv(file_path)
        print('Plotting')

        settings_column = [column for column in df.columns if column.startswith('settings')]

        for setting in settings_column:
            plt.figure(figsize=(12, 6))
            # sns.lineplot(data=df, x='time cycles', y='settings 1', hue='unit number')
            # sns.lineplot(data=df, x='time cycles', y='settings 2', hue='unit number')
            sns.lineplot(data=df, x='time cycles', y=setting, hue='unit number')

            plt.xlabel('Time Cycles')
            plt.ylabel(f'{setting} of the Turbofans')
            plt.title(f'{setting} Values over Time for each unit')
            plt.legend(title="Unit Number")
            plt.show()



def main():
    print('EDA Starts')
    file_path = 'clean_data/'
    analysis = DataAnalysis(file_path)
    analysis.plotSettingsToTime()



if __name__ == '__main__':
    main()




