import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.torch.autoencoder.autoencoder import Autoencoder, CombinedAutoencoderRegressionModel


def pretrain_autoencoders(autoencoders, data_list, num_epochs=50, batch_size=3, learning_rate=0.001, l2_strength=0.001):
    """
    Pretrain the autoencoders.

    Parameters:
    - autoencoders: List of autoencoders to pretrain
    - data_list: List of tensors, each corresponding to the input data for an autoencoder
    - num_epochs: Number of epochs to train each autoencoder (default: 50)
    - batch_size: Batch size for training (default: 64)
    - learning_rate: Learning rate for the optimizer (default: 0.001)

    Returns:
    - List of pretrained autoencoders
    """
    # Pretrain each autoencoder individually
    for i, (autoencoder, data) in enumerate(zip(autoencoders, data_list)):
        # Define loss function and optimizer for the current autoencoder
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        
        # Prepare DataLoader for the current dataset
        dataset = TensorDataset(data, data)  # Autoencoder target is the same as the input
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # print("Training autoencoder ", i)
        # Training loop for the current autoencoder
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_data, _ in dataloader:
                # Forward pass
                # print("Input shape: ", batch_data[0].shape)

                reconstructed = autoencoder(batch_data)
                # print("Reconstruction shape: ", reconstructed[0].shape)
                loss = criterion(reconstructed, batch_data)
                
                for param in autoencoder.parameters():
                    loss += l2_strength * torch.sum(param ** 2)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()
            
            # # Print average loss for the epoch every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #     avg_loss = epoch_loss / len(dataloader)
                # print(f'Autoencoder {i+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # print(f"Autoencoder {i+1} pretrained successfully.\n")

    print("All autoencoders pretrained.")
    return autoencoders




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_combined_autoencoder_regression(
    pretrained_autoencoders, 
    regression_layers,
    tensor_data, 
    target_tensor, 
    num_epochs=50, 
    batch_size=8, 
    learning_rate=0.001,
    l2_strength=0.0001,
    freeze_encoders=False
):
    """
    Trains a combined model consisting of multiple pretrained autoencoders and a regression model.
    
    Parameters:
    - pretrained_autoencoders: List of pretrained autoencoders
    - tensor_data: List of tensors, each corresponding to the input data for an autoencoder
    - target_tensor: Tensor containing the target values with shape [24, 1]
    - num_epochs: Number of epochs for training (default: 50)
    - batch_size: Batch size for training (default: 8)
    - learning_rate: Learning rate for the optimizer (default: 0.001)
    
    Returns:
    - trained combined model
    """
    


    # Instantiate the combined model
    encoded_dim_list = [encoder.encoder[-1].out_features for encoder in pretrained_autoencoders]
    combined_model = CombinedAutoencoderRegressionModel(pretrained_autoencoders, encoded_dim_list, regression_layers, freeze_encoders=freeze_encoders)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    # Prepare DataLoader
    dataset = TensorDataset(*tensor_data, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Extract the input tensors and target from the batch
            batch_data = batch[:-1]  # All except the last item
            batch_target = batch[-1].squeeze()  # The last item is the target
            
            # Forward pass
            outputs = combined_model(batch_data).squeeze()
            
            # print("Regression output shape: ", outputs.shape)
            # print("Target shape: ", batch_target.shape)
            loss = criterion(outputs, batch_target)
            
            for param in combined_model.parameters():
                loss += l2_strength * torch.sum(param ** 2)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Print average loss for the epoch every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     avg_loss = epoch_loss / len(dataloader)
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete for combined model.")
    
    return combined_model


def train_ensemble(X_list, y, encoder_layers, decoder_layers, regression_layers, 
                   pretrain_epochs=50, cm_epochs=50, ae_lr=0.001, cm_lr=0.001, ae_l2_strength=0.000001, 
                   cm_l2_strength=0.000001, sample_ids=None, scaler=MinMaxScaler, 
                   decoder_final_activation=nn.Sigmoid, freeze_encoders=False):

    loocv = LeaveOneOut()

    errors = {}
    filtered_data = X_list
    
    report_dict = {}
    
    
    autoencoder_report = {}
    
    
    for train_index, test_index in loocv.split(filtered_data[0]):
        
        test_subject = sample_ids[test_index[0]] if sample_ids is not None else str(test_index[0])

        print("-----------------------------------------------------------------")
        print(f"Running subject {test_subject}")
        
        
        
        filtered_X_train = [df.iloc[train_index] for df in filtered_data]
        filtered_X_test = [df.iloc[test_index] for df in filtered_data]
        target_train = y[train_index]
        target_test = y[test_index]
        
        if scaler is not None:
            X_scalers = [scaler() for _ in range(len(filtered_data))]
            y_scaler = scaler()
            scaled_X_train = [X_scalers[i].fit_transform(df) for i, df in enumerate(filtered_X_train)]
            scaled_X_test = [X_scalers[i].transform(df) for i, df in enumerate(filtered_X_test)]
            
            scaled_y_train = y_scaler.fit_transform(target_train.values.reshape(-1, 1))
            scaled_y_test = y_scaler.transform(target_test.values.reshape(-1, 1))
        else:
            scaled_X_train = [filtered_X_train[i].values for i in range(len(filtered_data))]
            scaled_X_test = [filtered_X_test[i].values for i in range(len(filtered_data))]
            scaled_y_train = target_train.values
            scaled_y_test = target_test.values
            
        
        tensor_data_train = [torch.tensor(df, dtype=torch.float32) for df in scaled_X_train]
        target_tensor_train = torch.tensor(scaled_y_train, dtype=torch.float32)
        
        tensor_data_test = [torch.tensor(df, dtype=torch.float32) for df in scaled_X_test]
        
        autoencoders = [Autoencoder(encoder_layers, decoder_layers, decoder_final_activation) for i in range(len(filtered_data))]

        # Pretrain the autoencoders
        pretrained_autoencoders = pretrain_autoencoders(
            autoencoders=autoencoders,  # List of autoencoders
            data_list=tensor_data_train,       # List of tensors, one for each autoencoder
            num_epochs=pretrain_epochs,               # Number of epochs for pretraining
            batch_size=3,               # Batch size for pretraining
            learning_rate= ae_lr,          # Learning rate
            l2_strength=ae_l2_strength           # L2 regularization strength
        )

        training_reconstructions = {}
        for sample_idx in range(len([tensor_data_train[0]])):
            
            
            training_reconstructions[sample_idx] = {}
            training_reconstructions[sample_idx]['inputs'] = filtered_X_train
            training_reconstructions[sample_idx]['inputs_scaled'] = scaled_X_train
            
            training_reconstructions[sample_idx]['reconstructions_scaled'] = [encoder(tensor_data_test[i][sample_idx]) for i, encoder in enumerate(pretrained_autoencoders)]

            if scaler is not None:
                training_reconstructions[sample_idx]['reconstructions_unscaled'] = [X_scalers[i].inverse_transform(encoder(tensor_data_test[i][sample_idx]).detach().numpy().reshape(1,-1)) for i, encoder in enumerate(pretrained_autoencoders)]
            else:
                training_reconstructions[sample_idx]['reconstructions_unscaled'] = training_reconstructions[sample_idx]['reconstructions_scaled']

        
        reconstruction_errors = []
        test_reconstructions = {}
        test_reconstructions['inputs'] = filtered_X_test
        test_reconstructions['inputs_scaled'] = scaled_X_test
        test_reconstructions['reconstructions_scaled'] = [encoder(tensor_data_test[i]) for i, encoder in enumerate(pretrained_autoencoders)]

        if scaler is not None:
            test_reconstructions['reconstructions_unscaled'] = [X_scalers[i].inverse_transform(encoder(tensor_data_test[i][sample_idx]).detach().numpy().reshape(1,-1)) for i, encoder in enumerate(pretrained_autoencoders)]
        else:
            test_reconstructions['reconstructions_unscaled'] = test_reconstructions['reconstructions_scaled']
        for i, encoder in enumerate(pretrained_autoencoders):
            reconstruction_error = torch.round(torch.abs(encoder(tensor_data_test[i]) - tensor_data_test[i]).mean(), decimals=5).detach().numpy()
            print("Reconstruction error: ", reconstruction_error)   
            reconstruction_errors.append(reconstruction_error)

        test_reconstructions['reconstruction_mean_abs_errors'] = reconstruction_errors
        
        average_reconstruction_error = np.mean(reconstruction_errors).round(decimals=5)
        
        combined_model = train_combined_autoencoder_regression(pretrained_autoencoders, regression_layers, 
                                                               tensor_data_train, target_tensor_train, 
                                                               num_epochs=cm_epochs, 
                                                               learning_rate=cm_lr, l2_strength=cm_l2_strength,
                                                               freeze_encoders=False)
        
        prediction = combined_model(tensor_data_test)
        if scaler is not None:
            prediction = y_scaler.inverse_transform(prediction.detach().numpy().flatten().reshape(-1, 1))
        else:
            prediction = prediction.detach().numpy().flatten().reshape(-1, 1)    
        error = np.abs(target_test.values - prediction)
        
        baseline_error = np.abs(target_test.values - np.mean(target_train))
        
        errors[str(test_index)] = error[0][0]
        print(f"Regression error {test_subject}: {np.round(errors[str(test_index)], 3)} || Baseline error: {baseline_error} || Reconstruction error: {average_reconstruction_error}")
        
        print("-----------------------------------------------------------------")
        
        report_dict[test_subject] = {}
        report_dict[test_subject]['ground_truth'] = target_test
        report_dict[test_subject]['prediction'] = prediction
        report_dict[test_subject]['error'] = error
        report_dict[test_subject]['baseline'] = np.mean(target_train)
        report_dict[test_subject]['baseline_error'] = baseline_error
        
        autoencoder_report[test_subject] = {}
        autoencoder_report[test_subject]['training_reconstructions'] = training_reconstructions
        autoencoder_report[test_subject]['test_reconstructions'] = test_reconstructions
        autoencoder_report[test_subject]['average_reconstruction_error'] = average_reconstruction_error
        autoencoder_report[test_subject]['reconstruction_errors'] = reconstruction_errors
        
    return report_dict, autoencoder_report
    