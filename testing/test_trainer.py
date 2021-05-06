from testing.data_loader import train_loader, test_loader
from deepness.trainer.trainer import Trainer



def test_trainer():
    train = train_loader
    test = test_loader
    print(f'this is the type: {type(train_loader)}')
    assert train, 'no train'
