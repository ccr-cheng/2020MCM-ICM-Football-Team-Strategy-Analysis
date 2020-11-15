import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

import time, sys

from transformerModel import TransformerEncoder
from data_utils import read_info, get_play_actions

INPUT_SIZE = 37
EMBED_SIZE = 128
HIDDEN_SIZE = 256
MAX_LENGTH = 64
NUM_LAYERS = 3
PAD_TOKEN = 0
MATCH_NUM = 30
PRINT_EVERY = 100

LR = 1e-4
N_EPOCHS = 200
CLIP_GRAD = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# '' is for Offside, we ignore Subsitition
event_names = ['<pad>', 'Goal kick', 'Air duel', 'Throw in', 'Head pass', 'Ground loose ball duel',
               'Simple pass', 'Launch', 'High pass', 'Touch', 'Ground defending duel', 'Hand pass',
               'Ground attacking duel', 'Foul', 'Free kick cross', 'Goalkeeper leaving line',
               '', 'Free Kick', 'Smart pass', 'Cross', 'Save attempt', 'Corner', 'Clearance',
               'Shot', 'Acceleration', 'Reflexes', 'Late card foul', 'Simulation',
               # 'Substitution' ,
               'Free kick shot', 'Protest', 'Hand foul', 'Penalty', 'Violent Foul',
               'Whistle', 'Out of game foul', 'Ball out of the field', 'Time lost foul']
event2idx = {name: idx for idx, name in enumerate(event_names)}

print('Reading data ...')
events = read_info('fullevents.csv')
# passing = read_info('passingevents.csv')
matches = read_info('matches.csv')
print('Data reading complete!')

model = TransformerEncoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                           PAD_TOKEN, MAX_LENGTH).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


def actions2tensor(actions):
    def events2tensor(action):
        types, coords = [], []
        for ev in action:
            ev_type = ev['EventSubType']
            if ev_type == 'Substitution':
                continue
            ev_coord = [ev['EventOrigin_x'], ev['EventOrigin_y']]
            if ev['EventDestination_x'] == '':
                ev_coord.extend([0, 0])
            else:
                ev_coord.extend([ev['EventDestination_x'], ev['EventDestination_y']])
            types.append(event2idx[ev_type])
            coords.append(ev_coord)
        types = torch.tensor(types, dtype=torch.long, device=device)
        coords = torch.tensor(coords, dtype=torch.float, device=device)
        return types, coords

    def get_team(action):
        return 1 if action[0]['TeamID'] == 'Huskies' else 0

    ev_types, ev_coords = zip(*(events2tensor(ac) for ac in actions))
    team_id = torch.tensor([get_team(ac) for ac in actions], dtype=torch.float, device=device)
    ev_types = pad_sequence(ev_types, padding_value=PAD_TOKEN)
    ev_coords = pad_sequence(ev_coords, padding_value=0)
    return ev_types, ev_coords, team_id


def match_loader(match_id):
    trg = 0
    for ma in matches:
        if ma['MatchID'] == match_id:
            trg = torch.tensor([ma['OwnScore'], ma['OpponentScore']],
                               dtype=torch.float, device=device)
            break
    a_match = [ev for ev in events if ev['MatchID'] == match_id]
    actions = get_play_actions(a_match, 3)
    if len(actions) > MAX_LENGTH:
        actions = actions[:MAX_LENGTH]
    return actions, trg


def train_epoch(cur_epoch):
    model.train()
    epoch_loss = 0

    for match_id in range(MATCH_NUM):
        match_id += 1
        actions, trg = match_loader(match_id)
        ev_types, ev_coords, team_id = actions2tensor(actions)

        optimizer.zero_grad()
        output = model(ev_types, ev_coords)
        own_score = torch.sum(output * team_id)
        oppo_score = torch.sum(output * (-team_id + 1))
        pred = torch.stack([own_score, oppo_score])
        loss = criterion(pred, trg)
        epoch_loss += loss.item()
        loss.backward()

        clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        # if match_id % PRINT_EVERY == 0:
        #     print(f'\nEpoch {cur_epoch}, match {match_id} / {MATCH_NUM}. Match loss: {loss:.4f}')
        #     print('Prediction: [{0[0]:.3f}, {0[1]:.3f}]'.format(pred))
        #     print(f'Target: {trg.tolist()}')
    return epoch_loss / MATCH_NUM


def valid(valid_num):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for match_id in range(38 - valid_num, 38):
            match_id += 1
            actions, trg = match_loader(match_id)
            ev_types, ev_coords, team_id = actions2tensor(actions)
            output = model(ev_types, ev_coords)
            own_score = torch.sum(output * team_id)
            oppo_score = torch.sum(output * (-team_id + 1))
            pred = torch.stack([own_score, oppo_score])
            loss = criterion(pred, trg)
            total_loss += loss.item()
    return total_loss / valid_num


def train():
    train_his = []
    valid_his = []
    for epoch in range(N_EPOCHS):
        start_epoch = time.time()
        train_loss = train_epoch(epoch + 1)
        valid_loss = valid(8)
        train_his.append(train_loss)
        valid_his.append(valid_loss)

        secs = int(time.time() - start_epoch)
        mins = secs // 60
        secs = secs % 60

        print(f'Epoch: {epoch + 1} | time in {mins} minutes, {secs} seconds')
        print(f'\tLoss: {train_loss:.4f}(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)')
        # model.saveModel('model.pt')

    with open('training.txt', 'w') as fout:
        data_list = [train_his, valid_his]
        names = ['Training Loss', 'Valid Loss']
        for data, name in zip(data_list, names):
            fout.write(f'\n{name}:\n')
            for num in data:
                fout.write(f'{num} ')


def evaluate(path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_utils import draw_pitch, draw_arrow

    modeldata = torch.load(path, map_location=device)
    print(f'Loading models from {path} ...')
    model.load_state_dict(modeldata['model_state'])
    model.eval()
    print('Loading models complete!\n')

    test_loss = valid(38)
    print(f'Testing loss: {test_loss:.3f}')

    match_id = 33
    actions, trg = match_loader(match_id)
    ev_types, ev_coords, team_id = actions2tensor(actions)
    output = model(ev_types, ev_coords)
    own_score = torch.sum(output * team_id)
    oppo_score = torch.sum(output * (-team_id + 1))
    print(f'Prediction: [{own_score:.2f}, {oppo_score:.2f}]')
    print(f'Target: [{trg[0]}, {trg[1]}]]')

    top_n = 10
    top_s, top_i = torch.topk(output * team_id, top_n)
    fig, ax = draw_pitch("#195905", "#faf0e6")
    handles = []
    for c, idx in enumerate(top_i):
        draw_arrow(actions[idx], ax, f'C{c}')
        line = Line2D([], [], color=f'C{c}', label=f'No. {c + 1}')
        handles.append(line)
    plt.legend(handles=handles, loc=2, bbox_to_anchor=(1, 1))
    plt.title(f'MatchID {match_id} top {top_n} scores', fontsize=20)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        train()
