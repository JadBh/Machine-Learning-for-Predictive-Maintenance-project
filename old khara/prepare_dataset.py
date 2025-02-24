import data_load_to_pd_df as dp
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import torch
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class Alpiq_Dataset():

    def __init__(self, dataset_root, unit = 'VG5',
                 load_training = True ,
                 load_synthetic = True, 
                 load_testing = False,
                 task = 'Conditional'):
        
        self.load_training = load_training
        self.load_testing = load_testing
        self.load_synthetic = load_synthetic

        self.task = task # conditional or joint distribution modeling 
        self.unit = unit

        if unit =='VG4':
            # note that from this call we get a dictionary with testing and training data, along with an info section using the 
            # parquet files!
            self.df = dp.RawDataset(dataset_root, unit, load_synthetic=False, load_training=True)
        else:
            self.df = dp.RawDataset(dataset_root, unit, load_synthetic=True, load_training=True)
        
        self._train_meas = self.df.data_dict["train"].measurements # extract training measurements
        self._train_info = self.df.data_dict["train"].info # extract training info
        self._test_meas = self.df.data_dict["test"].measurements # extract test measurements
        self._test_info = self.df.data_dict["test"].info # extract test info

        self._train_meas = self._train_meas.reset_index()
        self.timeline  = self._train_meas["index"] # save this for plotting anything
        self._train_meas = self._train_meas.drop(columns = ["index"])# drop the timestep as we do not need it

        self._test_meas = self._test_meas.reset_index()
        self.timeline  = self._test_meas["index"] # save this for plotting anything
        self._test_meas = self._test_meas.drop(columns = ["index"])# drop the timestep as we do not need it

        self._train_meas_filtered_pump,self._train_meas_filtered_turbine,self._train_meas_filtered_scm = self.remove_unwanted_states() # keep the steady state only
        
        # self.control_signal_data, self.output_data = self.divide_into_functionality(self._train_meas_filtered_pump) # divide based on control inputs or output measurements


    def remove_unwanted_states(self): # remove transcients and machine OFF

        if self.unit  == 'VG5' or self.unit == 'VG6':
            train_meas_pump = self._train_meas [
                                            ((self._train_meas['equilibrium_pump_mode'] == True) & 
                                                (self._train_meas['dyn_only_on'] == False))]
            
            train_meas_turbine = self._train_meas[
                ((self._train_meas['equilibrium_turbine_mode'] == True) & 
                 (self._train_meas['dyn_only_on'] == False))]

            train_meas_short_circuit = self._train_meas[
                ((self._train_meas['equilibrium_short_circuit_mode'] == True) &
                 (self._train_meas['dyn_only_on'] == False))]
            return train_meas_pump, train_meas_turbine, train_meas_short_circuit
            
        else: # in case of VG4, we do not have short circuit modes

            train_meas_pump = self._train_meas [
                                            ((self._train_meas['equilibrium_pump_mode'] == True) & 
                                                (self._train_meas['dyn_only_on'] == False))]
            
            train_meas_turbine = self._train_meas[
                ((self._train_meas['equilibrium_turbine_mode'] == True) & 
                 (self._train_meas['dyn_only_on'] == False))]
        
            return train_meas_pump, train_meas_turbine, None

    def divide_into_functionality(self,df): # change the function to do it for every mini dataset!
        """
        Here mainly if we want to train a forecasting model, we need to divide the signals based on whether they are 
        1- Control Signal
        2- Output Signal 
        3- Input Signal (since we only have 1 and its a common signal with the output we will forget about this part )
        """

        # Identify control signals and output features
        control_signals = self._train_info[self._train_info['control_signal'] == True]['attribute_name'].tolist()
        outputs = self._train_info[self._train_info['output_feature'] == True]['attribute_name'].tolist()
        
        columns = df.columns

        control_list = []
        output_list = []

        for signal in control_signals:
            if signal in columns:
                control_list.append(signal)

        for signal in outputs:
            if signal in columns:
                output_list.append(signal)
                
        # Extract corresponding columns from measurements
        control_signal_data = df[control_list]
        output_data = df[output_list]

        return control_signal_data, output_data
    
    @staticmethod
    def add_group_id(df):

        df = df.copy()
        df['index_diff'] = df['index'] - df['index'].shift(1).fillna(df['index'].iloc[0])
        # print(df['index'].to_numpy())


    # Create a new group whenever the difference exceeds 1
        df['group_id'] = (df['index_diff'] > 10).cumsum()
        # Drop the helper column if not needed
        df = df.drop(columns=['index_diff'])

        """length of some groups: 
        0: 0-1880 
        1: 1881-2335 (454)
        10: 6517- 7453 (936)
        we have 219 windows
        use the commented print below to know
        """
        # print((df[df['group_id']==10]))


        return df


class SlidingWindowDataset(Dataset):
    def __init__(self, data_frame, window=50, stride=10, horizon=1, device='cpu'):
        """Sliding window dataset with RUL label

        Args:
            dataframe (pd.DataFrame): dataframe containing scenario descriptors and sensor reading
            window (int, optional): sequence window length. Defaults to 50.
            stride (int, optional): data stride length. Defaults to 1.
        """
        # define the windowing parameters 
        self.window = window # time window length
        self.stride = stride # stride length when applying widowing
  

        #note that one day is 2880 timesteps
        self.X = np.array(data_frame[XS_VAR].head(100000000).values).astype(np.float32)

        data_frame = data_frame.reset_index()
        # data_frame = Alpiq_Dataset.process_timeline_column(data_frame)
        data_frame = Alpiq_Dataset.add_group_id(data_frame)


        data_frame['window'] = (data_frame.index // window).astype(int)
        # Group by the window index and count the number of samples in each window
        window_counts = data_frame.groupby('window').size()

        self.indices = torch.from_numpy(self._get_indices(window_counts)).to(device) 
        self.group_id = data_frame['group_id'].to_numpy()
        self.indexes = data_frame['index'].to_numpy()
        # print(self.indexes)

        # print(self.group_id)


    def _get_indices(self, window_counts):
        counts = window_counts.to_numpy()
        idx_list = []
        for i, w_count in enumerate(counts):  # w_count is the number of sample in each window
            if i ==0:
                w_start = sum(counts[:i]) 
            else:
                w_start = sum(counts[:i]) - self.stride
            # print(f"w_count: {w_count}, w_start: {w_start}")
            w_end = w_start + w_count
            if w_end < len(self.X):
                idx_list += [_ for _ in np.arange(w_start, w_end + 1, self.stride)]

        return np.asarray([(idx, idx + self.window) for idx in idx_list]) 
            
    def __len__(self):# number of windowed segments
        return len(self.indices)


    def __getitem__(self, i):
        """ instead of padding with zeros/mean skip the final window if it doesnt have the same size as others"""
        # Get the start and end indices of the window
        i_start, i_stop = self.indices[i]
        # indices = 
        # Get the group_id for the starting index
        current_group_id = self.group_id[i_start]
        # print(len(self.group_id))
        indice = np.where(self.indexes ==(i_stop-1))[0]
        print(indice)
        # Check if the entire window is within the same group
            # Adjust the stop index to the last index of the current group
        if indice.size == 0 or self.group_id[indice[0]] != current_group_id:
            return None 
        else:
            
            x = self.X[i_start:i_stop, :]

        # Return the transposed window for PyTorch compatibility
        return torch.tensor(x).permute(1, 0)


def create_datasets(df, window_size,train = True, device='cpu'):
    df = df.drop(columns=['equilibrium_short_circuit_mode','machine_on', 'machine_off', 'turbine_mode','all', 'equilibrium_turbine_mode', 'dyn_only_on', "short_circuit_mode",'pump_mode', 'equilibrium_pump_mode' ])
    # print(type(df))
    dataset = SlidingWindowDataset(df, window=window_size)

    # normalizing features
    scaler = MinMaxScaler()

    if train:
        dataset.X = scaler.fit_transform(dataset.X)
    else: 
        dataset.X = scaler.transform(dataset.X)


    # convert numpy array to tensors
    datasets = [dataset]
    for d in datasets:
        d.X = torch.from_numpy(d.X).to(device)
    
    # print(type(datasets[0]))
    return datasets

def create_data_loaders(datasets, batch_size=256, val_split=0.2):
    # fixed seed for data splits for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    d_train = datasets[0]
    # print(len(d_train))
    dataset_size = len(d_train)
    indices = list(range(dataset_size))
    # split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices= indices
    train_sampler = SubsetRandomSampler(train_indices)


    train_loader = DataLoader(d_train, batch_size=batch_size, sampler=train_sampler)


    # d_info = f"train_size: {len(train_indices)}\t"

    # print(d_info)
    return train_loader
# STILL NEED TO DO A TIME WINDOWING 
# STILL NEED TO DO A PLOTTING FUNCTIONS TO VISUALIZE 
# STILL NEED TO DO A FUNCTION TO MAKE IT INTO PYTORCH DATALOADERS


root = Path(r"./Dataset")
A1 = Alpiq_Dataset(root, 'VG5', True, False, True)
XS_VAR = A1._train_info[A1._train_info['signal_type'] == 'Measurement']['attribute_name'].tolist()

def main_dataset():
    DATASETS = create_datasets(A1._train_meas_filtered_pump, window_size=50, train= True)   # returns sliding window class, whose elements [i] are each a sliding window of 50

    for i, x in enumerate(DATASETS):
        print("length x is: ", len(x))
        print("x element type is: ", type(x[10]))
        print("x element shape is: ", x[10].shape)

    
    X = create_data_loaders(DATASETS) 
    print(next(iter(X)))

    for x in X:
        print('hi')
    # divide into operating modes 
    
    # for every operating mode, create a sliding window dataset
    # create for every dataset a dataloader 
    # Train ya kbir

    # print(X)
    return 0


if __name__ == "__main__":
    main_dataset()

