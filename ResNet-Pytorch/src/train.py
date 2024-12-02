import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
# assertNotEqual(csv_path, '', 'Could not locate the data.csv file')
data = pd.read_csv(csv_path, sep=';')

train, val = train_test_split(data, test_size=0.2,random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
batch_size=64
train_set=t.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=batch_size,  shuffle=True)
val_set=t.utils.data.DataLoader(ChallengeDataset(val, 'val'), batch_size=batch_size, shuffle=False)

# create an instance of our ResNet model
# TODO
model = model.ResNet()


# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
loss_fun=t.nn.BCELoss()
# optimizer=t.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = t.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

trainer=Trainer(model=model, crit=loss_fun, optim=optimizer, train_dl=train_set, val_test_dl=val_set, cuda=True, early_stopping_patience=5)

# go, go, go... call fit on trainer
# res = #TODO
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')