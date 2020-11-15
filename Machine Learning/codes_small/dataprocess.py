from plot_utils import draw_pitch, plot_kde_events_on_field
from plot_utils import plot_passing, passing_network
from data_utils import read_info, get_play_actions, get_invasion_index

print('Reading data ...')
events = read_info('fullevents.csv')
passing = read_info('passingevents.csv')
match = read_info('matches.csv')
print('Data reading complete!')
MATCH_NUM = 38


def calculate_connectivity(path='connectivity.txt'):
    print('Calculating connectivity ...')

    H_conn, H_cent = [], []
    O_conn, O_cent = [], []
    for match_id in range(MATCH_NUM):
        match_id += 1
        a_match = [ev for ev in passing if ev['MatchID'] == match_id]
        (a1, b1), (a2, b2) = passing_network(a_match)
        H_conn.append(a1)
        H_cent.append(b1)
        O_conn.append(a2)
        O_cent.append(b2)
    with open(path, 'w') as fout:
        data_list = [H_conn, H_cent, O_conn, O_cent]
        names = ['Huskies connectivity', 'Huskies centrality',
                 'Opponent connectivity', 'Opponent centrality']
        for data, name in zip(data_list, names):
            fout.write(f'\n{name}:\n')
            for num in data:
                fout.write(f'{num} ')

    print(f'Calculation complete! Output to {path}')


def calculate_indices(path='indices.txt'):
    print('Calculating invasion and acceleration indices ...')

    H_invasion, H_accel = [], []
    O_invasion, O_accel = [], []
    for match_id in range(MATCH_NUM):
        match_id += 1
        a_match = [ev for ev in events if ev['MatchID'] == match_id]
        invasion, accel = get_invasion_index(a_match)
        H_invasion.append(invasion[0])
        O_invasion.append(invasion[1])
        H_accel.append(accel[0])
        O_accel.append(accel[1])

    with open(path, 'w') as fout:
        data_list = [H_invasion, H_accel, O_invasion, O_accel]
        names = ['Huskies Invasion', 'Huskies Acceleration',
                 'Opponent Invasion', 'Opponent Acceleration']
        for data, name in zip(data_list, names):
            fout.write(f'\n{name}:\n')
            for num in data:
                fout.write(f'{num} ')

    print(f'Calculation complete! Output to {path}')


def calculate_match_indices(match_id, path='match_indices.txt'):
    print('Calculating invasion and acceleration indices ...')

    a_match = [ev for ev in events if ev['MatchID'] == match_id]
    list_inv, list_acc = get_invasion_index(a_match, reduce=False)
    with open(path, 'w') as fout:
        data_list = [list_inv[0], list_acc[0], list_inv[1], list_acc[1]]
        names = ['Huskies Invasion', 'Huskies Acceleration',
                 'Opponent Invasion', 'Opponent Acceleration']
        for data, name in zip(data_list, names):
            fout.write(f'\n{name}:\n')
            for num in data:
                fout.write(f'{num} ')

    print(f'Calculation complete! Output to {path}')


def action_count(path='filter_len.txt'):
    print('Calculating action length ...')

    cnt = []
    for filter_len in range(1, 8):
        length = 0
        for match_id in range(MATCH_NUM):
            match_id += 1
            a_match = [ev for ev in events if ev['MatchID'] == match_id]
            length += len(get_play_actions(a_match, filter_len))
        cnt.append(length / MATCH_NUM)
    with open(path, 'w') as fout:
        fout.write('Filter Length\n')
        for num in cnt:
            fout.write(f'{num:.1f} ')

    print(f'Calculation complete! Output to {path}')


# plot_passing(a_match, match_id)
# draw_pitch("#195905", "#faf0e6")
# plot_kde_events_on_field(events)
# passing_network(a_match)

# phases = cut_phase(a_match)
# print(len(phases))
# for p in phases:
#     print(p[-1]['EventSubType'])

if __name__ == '__main__':
    # calculate_connectivity()
    calculate_indices()
    calculate_match_indices(15)
    # action_count()
    pass
