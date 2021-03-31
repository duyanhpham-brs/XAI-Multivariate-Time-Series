import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, scheduler, dataloaders, datasets_size, num_epochs=25):
    since = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    train_loss = []
    test_loss = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds == labels.data))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_corrects.double() / datasets_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Correct: {}'.format(
                phase, epoch_loss, epoch_acc, running_corrects.double()))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)

        plt.plot(np.array(train_loss))
        plt.plot(np.array(test_loss))
        plt.show()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
