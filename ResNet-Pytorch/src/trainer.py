import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

from sklearn.metrics import precision_score, recall_score
import os
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        
        self._optim.zero_grad()
        # self._model.zero_grad()
        output=self._model(x)
        loss=self._crit(output, y.float())
        loss.backward()
        self._optim.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        
        predictions=self._model(x)
        loss=self._crit(predictions, y.float())
        return loss.item(), predictions
    

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        
        self._model.train(True)
        total_loss=0.0
        for batch in self._train_dl:
            inputs, labels = batch
            if self._cuda:
                # batch = batch.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()
            
            total_loss += self.train_step(inputs, labels)
        avg_loss = total_loss / len(self._train_dl.dataset)
        return avg_loss

    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO

        self._model.eval()
        total_loss = 0.0
        correct = 0
        with t.no_grad():
            for batch in self._val_test_dl:
                inputs, labels = batch
                if self._cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                loss, predictions = self.val_test_step(inputs, labels)
                total_loss+=loss
                predictions[predictions >=  0.5] = 1
                predictions[predictions <  0.5] = 0
                correct += (predictions == labels).sum().item()

        avg_loss = total_loss / len(self._val_test_dl.dataset)
        
        accuracy = 100 * correct / (len(self._val_test_dl.dataset)*2)
        print("Accuracy: {}%".format(accuracy))

        ##################################################################################################
        # labels_np = labels.cpu().numpy()
        # predictions_np = predictions.cpu().numpy()

        # threshold = 0.5
        # predictions_binary = (predictions_np >= threshold).astype(int)

        # labels_binary = labels_np.astype(int)

        # # Check shapes
        # assert labels_binary.shape == predictions_binary.shape, "Shapes of labels and predictions must match."

        # # Calculate precision, recall, and F1 score with the 'samples' average method
        # precision = precision_score(labels_binary, predictions_binary, average='samples')
        # recall = recall_score(labels_binary, predictions_binary, average='samples')
        # f1 = f1_score(labels_binary, predictions_binary, average='samples')

        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1 Score: {f1}")
        ##################################################################################################

        return avg_loss


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_losses=[]
        val_losses=[]
        epoch_counter=1
        best_val_loss = 1000000.
        early_stopping_patience = self._early_stopping_patience
        try:
            while True:
        
                # stop by epoch number
                # train for a epoch and then calculate the loss and metrics on the validation set
                # append the losses to the respective lists
                # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
                # check whether early stopping should be performed using the rearly stopping criterion and stop if so
                # return the losses for both training and validation
            #TODO
                if epochs > 0 and epoch_counter > epochs:
                    break
                train_loss = self.train_epoch()
                val_loss = self.val_test()
                print("epoch:{}, test error: {}, val error: {}".format(epoch_counter, train_loss, val_loss))
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    # only to manage limited disk quota in cip machines
                    if epoch_counter % 10 == 0:
                        for file in os.scandir("./checkpoints/"):
                            if file.name.endswith(".ckp"):
                                os.unlink(file.path) 

                    self.save_checkpoint(epoch_counter)
                    best_val_loss = val_loss
                    if self._early_stopping_patience > 0:
                        early_stopping_patience = self._early_stopping_patience
                else:
                    print("validation error increase")
                    if self._early_stopping_patience > 0:
                        early_stopping_patience-=1
                        if early_stopping_patience == 0:
                            break
                epoch_counter+=1
        except Exception as e:
            print(f"Error processing batch: {e}")
        print("end of training...")
        return train_losses, val_losses

