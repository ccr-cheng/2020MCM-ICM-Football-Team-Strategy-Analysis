import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

import time
import argparse

from transformerModel import TransformerEncoder
from data_utils import read_data, matchify

parser = argparse.ArgumentParser(description='An implementation of the Transformer model')
parser.add_argument('--train-set', type=str, default='World_Cup', help='training set')
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--num-epoch', type=int, default=20, help='number of training epochs')
parser.add_argument('--num-layer', type=int, default=3, help='number of layers in encoder and decoder')
parser.add_argument('--embed-size', type=int, default=128, help='embedding size (d_model)')
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size (in the feedforward layers)')
parser.add_argument('--max-length', type=int, default=64, help='max length to trim the dataset')
parser.add_argument('--clip-grad', type=float, default=1.0, help='parameter clipping threshold')
parser.add_argument('--print-every', type=int, default=100, help='print training procedure every number of batches')
parser.add_argument('--save-path', type=str, default='model.pt', help='model path for saving')
parser.add_argument('--log-path', type=str, default='training.txt', help='log path')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint for resuming training')
args = parser.parse_args()

INPUT_SIZE = 37
EMBED_SIZE = args.embed_size
HIDDEN_SIZE = args.hidden_size
MAX_LENGTH = args.max_length
NUM_LAYERS = args.num_layer
PRINT_EVERY = args.print_every
PAD_TOKEN = 0

LR = args.lr
N_EPOCHS = args.num_epoch
CLIP_GRAD = args.clip_grad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

event_names = ['<pad>', 'Goal kick', 'Air duel', 'Throw in', 'Head pass', 'Ground loose ball duel',
               'Simple pass', 'Launch', 'High pass', 'Touch', 'Ground defending duel', 'Hand pass',
               'Ground attacking duel', 'Foul', 'Free kick cross', 'Goalkeeper leaving line',
               '', 'Free Kick', 'Smart pass', 'Cross', 'Save attempt', 'Corner', 'Clearance',
               'Shot', 'Acceleration', 'Reflexes', 'Late card foul', 'Simulation',
               'Free kick shot', 'Protest', 'Hand foul', 'Penalty', 'Violent Foul',
               'Whistle', 'Out of game foul', 'Ball out of the field', 'Time lost foul']
event2idx = {name: idx for idx, name in enumerate(event_names)}
eventid2idx = {0: 0, 60: 16, 10: 2, 11: 12, 12: 10, 13: 5, 20: 13, 21: 30, 22: 26,
               23: 34, 24: 29, 25: 27, 26: 36, 27: 32, 30: 21, 31: 17, 32: 14, 33: 28,
               34: 1, 35: 31, 36: 3, 40: 15, 50: 35, 51: 33, 70: 24, 71: 22, 72: 9,
               80: 19, 81: 11, 82: 4, 83: 8, 84: 7, 85: 6, 86: 18, 90: 25, 91: 20, 100: 23}

print('Reading data ...')
batches = matchify(*read_data(args.train_set))
print('Data reading complete!')

MATCH_NUM = len(batches)
VALID_NUM = 20
TRAIN_NUM = MATCH_NUM - VALID_NUM
train_batch = batches[0:TRAIN_NUM]
valid_batch = batches[TRAIN_NUM:]

model = TransformerEncoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                           PAD_TOKEN, MAX_LENGTH).to(device)
criterion = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


def actions2tensor(actions, team):
    def events2tensor(action):
        types, coords = [], []
        for ev in action:
            pos = ev['positions']
            if len(pos) == 1:
                ev_coord = [pos[0]['x'], pos[0]['y'], 0, 0]
            else:
                ev_coord = [pos[0]['x'], pos[0]['y'],
                            pos[1]['x'], pos[1]['y']]
            types.append(event2idx[ev['subEventName']])
            coords.append(ev_coord)
        types = torch.tensor(types, dtype=torch.long, device=device)
        coords = torch.tensor(coords, dtype=torch.float, device=device)
        return types, coords

    def get_team(action):
        return 1 if action[0]['teamId'] == int(team) else 0

    ev_types, ev_coords = zip(*(events2tensor(ac) for ac in actions))
    team_id = torch.tensor([get_team(ac) for ac in actions], dtype=torch.float, device=device)
    ev_types = pad_sequence(ev_types, padding_value=PAD_TOKEN)
    ev_coords = pad_sequence(ev_coords, padding_value=0)
    return ev_types, ev_coords, team_id


def train_epoch(cur_epoch):
    model.train()
    epoch_loss = 0

    for idx, (actions, trg, team) in enumerate(train_batch):
        ev_types, ev_coords, team_id = actions2tensor(actions, team)

        optimizer.zero_grad()
        output = model(ev_types, ev_coords)
        own_score = torch.sum(output * team_id)
        oppo_score = torch.sum(output * (-team_id + 1))
        pred = torch.stack([own_score, oppo_score])
        trg = torch.tensor(trg, dtype=torch.float, device=device)
        loss = criterion(pred, trg)
        epoch_loss += loss.item()
        loss.backward()

        clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        if idx % PRINT_EVERY == 0:
            print(f'\nEpoch {cur_epoch}, match {idx} / {TRAIN_NUM}. Match loss: {loss:.4f}')
            print('Prediction: [{0[0]:.3f}, {0[1]:.3f}]'.format(pred))
            print(f'Target: {trg.tolist()}')
    return epoch_loss / TRAIN_NUM


def valid():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for actions, trg, team in valid_batch:
            ev_types, ev_coords, team_id = actions2tensor(actions, team)
            output = model(ev_types, ev_coords)
            own_score = torch.sum(output * team_id)
            oppo_score = torch.sum(output) - own_score
            pred = torch.tensor([own_score, oppo_score], dtype=torch.float, device=device)
            trg = torch.tensor(trg, dtype=torch.float, device=device)
            loss = criterion(pred, trg)
            total_loss += loss.item()
    return total_loss / VALID_NUM


def train():
    if args.checkpoint != '':
        path = args.checkpoint
        print(f'Loading checkpoint from {path} ...')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        print(f'Checkpoint loading complete!\n')

    train_his, valid_his = [], []
    for epoch in range(N_EPOCHS):
        start_epoch = time.time()
        train_loss = train_epoch(epoch + 1)
        valid_loss = valid()
        train_his.append(train_loss)
        valid_his.append(valid_loss)

        secs = int(time.time() - start_epoch)
        mins = secs // 60
        secs = secs % 60

        print(f'Epoch: {epoch + 1} | time in {mins} minutes, {secs} seconds')
        print(f'\tLoss: {train_loss:.4f}(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)')
        model.saveModel('model.pt')

    path = args.log_path
    with open(path, 'w') as fout:
        data_list = [train_his, valid_his]
        names = ['Training Loss', 'Valid Loss']
        for data, name in zip(data_list, names):
            fout.write(f'\n{name}:\n')
            for num in data:
                fout.write(f'{num} ')
    print(f'Training process saved to {path}!')


if __name__ == '__main__':
    train()
