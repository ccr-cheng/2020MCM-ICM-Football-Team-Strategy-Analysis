import numpy as np
import json


def read_data(nation, path='./data'):
    nations = ['Italy', 'England', 'Germany', 'France', 'Spain',
               'European_Championship', 'World_Cup']
    if nation not in nations:
        raise ValueError('Incorrect nation')
    print(f'Reading {nation} ...')
    with open(path + '/events/events_%s.json' % nation) as json_data:
        events = json.load(json_data)
    with open(path + '/matches/matches_%s.json' % nation) as json_data:
        matches = json.load(json_data)
    print(f'{nation} reading complete!')
    return events, matches


def matchify(events, matches):
    def get_score(match_id):
        for ma in matches:
            if ma['wyId'] == match_id:
                td = ma['teamsData']
                ids = [k for k in td.keys()]
                return (td[ids[0]]['score'], td[ids[1]]['score']), ids[0]

    batch = []
    last_event = 0
    cur_match = events[0]['matchId']
    for idx in range(len(events)):
        match_id = events[idx]['matchId']
        if match_id != cur_match:
            score, team_id = get_score(cur_match)
            actions = get_play_actions(events[last_event:idx])
            batch.append((actions, score, team_id))
            last_event = idx
            cur_match = match_id
    else:
        score, team_id = get_score(cur_match)
        actions = get_play_actions(events[last_event:])
        batch.append((actions, score, team_id))

    return batch


def get_play_actions(events, filter_len=5):
    """
    Given a list of events occuring during a game, it splits the events
    into play actions using the following principle:

    - an action begins when a team gains ball possession
    - an action ends if one of three cases occurs:
    -- there is interruption of the match, due to: 1) end of first half or match; 2) ball
    out of the field 3) offside 4) foul

    """
    INTERRUPTION = 5
    FOUL = 2
    OFFSIDE = 6
    DUEL = 1
    SAVE_ATTEMPT = 91
    REFLEXES = 90
    PENALTY = 35

    END_OF_GAME_EVENT = {
        'eventName': -1,
        'eventSec': 7200,
        'id': -1,
        'matchId': -1,
        'matchPeriod': 'END',
        'playerId': -1,
        'positions': [],
        'subEventName': -1,
        'tags': [],
        'teamId': -1
    }

    START_OF_GAME_EVENT = {
        'eventName': -2,
        'eventSec': 0,
        'id': -2,
        'matchId': -2,
        'matchPeriod': 'START',
        'playerId': -2,
        'positions': [],
        'subEventName': -2,
        'tags': [],
        'teamId': -2
    }

    def is_interruption(event, current_half):
        event_id, match_period = event['eventName'], event['matchPeriod']
        if event_id in [INTERRUPTION, FOUL, OFFSIDE] or match_period != current_half or event_id == -1:
            return True
        return False

    def is_shot(event):
        event_id = event['eventName']
        return event_id == 10

    def is_save_attempt(event):
        return event['subEventName'] == SAVE_ATTEMPT

    def is_reflexes(event):
        return event['subEventName'] == REFLEXES

    def is_duel(event):
        return event['eventName'] == DUEL

    def is_ball_lost(event, previous_event):
        if event['teamId'] != previous_event['teamId'] and previous_event['teamId'] != -2 and event['eventName'] != 1:
            return True

        return False

    def is_penalty(event):
        return event['subEventName'] == PENALTY

    def add_actions(action):
        if len(action) >= filter_len:
            play_actions.append(action)

    ## add a fake event representing the start and end of the game
    events.insert(0, START_OF_GAME_EVENT)
    events.append(END_OF_GAME_EVENT)

    play_actions = []
    time, index, current_action, current_half = 0.0, 1, [], '1H'
    previous_event = events[0]
    while index < len(events) - 2:
        current_event = events[index]

        # if the action stops by an game interruption
        if is_interruption(current_event, current_half):
            current_action.append(current_event)
            add_actions(current_action)
            current_action = []

        elif is_penalty(current_event):
            next_event = events[index + 1]

            if is_save_attempt(next_event) or is_reflexes(next_event):
                index += 1
                current_action.append(current_event)
                current_action.append(next_event)
                add_actions(current_action)
                current_action = []
            else:
                current_action.append(current_event)

        elif is_shot(current_event):
            next_event = events[index + 1]

            if is_interruption(next_event, current_half):
                index += 1
                current_action.append(current_event)
                current_action.append(next_event)
                add_actions(current_action)
                current_action = []

            ## IF THERE IS A SAVE ATTEMPT OR REFLEXES; GO TOGETHER
            elif is_save_attempt(next_event) or is_reflexes(next_event):
                index += 1
                current_action.append(current_event)
                current_action.append(next_event)
                add_actions(current_action)
                current_action = []

            else:
                current_action.append(current_event)
                add_actions(current_action)
                current_action = []

        elif is_ball_lost(current_event, previous_event):
            current_action.append(current_event)
            add_actions(current_action)
            current_action = [current_event]

        else:
            current_action.append(current_event)

        current_half = current_event['matchPeriod']
        index += 1

        if not is_duel(current_event):
            previous_event = current_event

    events.remove(START_OF_GAME_EVENT)
    events.remove(END_OF_GAME_EVENT)

    return play_actions


def cut_phase(events, min_len=3):
    phases = []
    cur_possess = events[0]['TeamID']
    cur_phase = []
    for ev in events:
        if ev['TeamID'] == cur_possess:
            cur_phase.append(ev)
        else:
            if len(cur_phase) >= min_len:
                phases.append(cur_phase)
            cur_phase = []
            cur_possess = ev['TeamID']
    return phases


def get_datadriven_weight(position, normalize=True):
    """
    Get the probability of scoring a goal given the position of the field where
    the event is generated.

    Parameters
    ----------
    position: tuple
        the x,y coordinates of the event

    normalize: boolean
        if True normalize the weights
    """
    weights = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 2.00000000e+00, 2.00000000e+00,
                         0.00000000e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 8.00000000e+00, 1.10000000e+01,
                         1.00000000e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         4.00000000e+00, 4.00000000e+01, 1.28000000e+02,
                         7.00000000e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         9.00000000e+00, 1.01000000e+02, 4.95000000e+02,
                         4.83000000e+02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         6.00000000e+00, 9.80000000e+01, 5.60000000e+02,
                         1.12000000e+03],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         8.00000000e+00, 9.30000000e+01, 5.51000000e+02,
                         7.82000000e+02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         3.00000000e+00, 6.70000000e+01, 3.00000000e+02,
                         2.30000000e+02],
                        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         1.00000000e+00, 1.30000000e+01, 3.20000000e+01,
                         1.10000000e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         1.00000000e+00, 2.00000000e+00, 2.00000000e+00,
                         2.00000000e+00],
                        [1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
                         0.00000000e+00]])

    x, y = position
    if x == 100.0:
        x = 99.9
    if y == 100.0:
        y = 99.9

    w = weights[int(y / 10)][int(x / 10)]
    if normalize:  # normalize the weights
        w = w / np.sum(weights)
    return w


def get_weight(position):
    """
    Get the probability of scoring a goal given the position of the field where
    the event is generated.

    Parameters
    ----------
    position: tuple
        the x,y coordinates of the event
    """
    x, y = position

    # 0.01
    if 65 <= x <= 75:
        return 0.01

    # 0.5
    if (75 < x <= 85) and (15 <= y <= 85):
        return 0.5
    if x > 85 and (15 <= y <= 25) or (75 <= y <= 85):
        return 0.5

    # 0.02
    if x > 75 and (y <= 15 or y >= 85):
        return 0.02

    # 1.0
    if x > 85 and (40 <= y <= 60):
        return 1.0

    # 0.8
    if x > 85 and (25 <= y <= 40 or 60 <= y <= 85):
        return 0.8

    return 0.0
