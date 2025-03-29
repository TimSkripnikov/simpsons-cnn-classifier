from Simpsons_CNN import Cnn, NetWork
import torch
import torch.nn as nn
import torch.optim as optim

def main():
   
    model = Cnn(n_classes=42)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    
    trainer = NetWork(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_dir='/home/a.skripnikov/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset',
        save_name='simpsons_model',
        epochs=15,
        batch_size=64,
        val_size=0.2
    )

    trainer.train()

if __name__ == "__main__":
    main()