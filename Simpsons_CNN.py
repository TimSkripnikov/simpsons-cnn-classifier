import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report

class Cnn(nn.Module):
    def __init__(self, n_classes=42):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = F.interpolate(x, size=(4, 4), mode='bilinear')
        x = x.view(x.size(0), 4 * 4 * 512)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


class DatasetLoader(Dataset):
    def __init__(self, files, mode, rescale_size=96):
        super().__init__()
        self.data_modes = ['train', 'val']
        self.files = sorted(files)
        self.mode = mode
        self.rescale_size = rescale_size

        if self.mode not in self.data_modes:
            print(f"{self.mode} is not correct; correct modes: {self.data_modes}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()
        self.labels = [path.parent.name for path in self.files]
        self.label_encoder.fit(self.labels)

        with open('label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)

    def __getitem__(self, index):
        x, size = self.load_sample(self.files[index])

        transforms_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-7, 7)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transforms_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.mode == 'train':
            x = transforms_train(x)
        else:
            x = transforms_val(x)
            
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        
        return x, y

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        image = image.resize((self.rescale_size, self.rescale_size))
        return image, image.size


class NetWork:

    def __init__(self, model, optimizer, criterion, scheduler, train_dir, 
                 save_name, use_scheduler=True, epochs=25, batch_size=64, val_size=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_dir = Path(train_dir)
        self.file_name = save_name
        self.use_scheduler = use_scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.history = []
        
        self.val_dataset = None
        self.train_dataset = None
        
        self.get_dataset()

    def get_dataset(self):
        train_val_files = sorted(list(self.train_dir.rglob('*.jpg')))
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=self.val_size,
            stratify=[path.parent.name for path in train_val_files]
        )
        
        self.train_dataset = DatasetLoader(train_files, mode='train')
        self.val_dataset = DatasetLoader(val_files, mode='val')

    def fit_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_size += inputs.size(0)
            
        train_loss = running_loss / processed_size
        train_acc = running_corrects.cpu().numpy() / processed_size
        return train_loss, train_acc

    def eval_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        report = classification_report(all_labels, all_preds, output_dict=True)
        class_report = pd.DataFrame(report).transpose()  
    
   
        class_report.to_csv('class_report.csv', mode='a', header=not pd.io.common.file_exists('class_report.csv'))
    

        return val_loss, val_acc, val_precision, val_recall, val_f1

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.fit_epoch(train_loader)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.eval_epoch(val_loader)
            
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            self.history.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1
            })
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
            
            # if val_f1 > best_val_f1:
            #     best_val_f1 = val_f1
            #     torch.save(self.model.state_dict(), f"{self.file_name}_best.pth")
            #     print("Saved new best model!")
        
        torch.save(self.model.state_dict(), f"{self.file_name}_final.pth")
        return best_val_f1

    