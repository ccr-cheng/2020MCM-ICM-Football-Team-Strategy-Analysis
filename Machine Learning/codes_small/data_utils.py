import numpy as np
import csv

int_type = {'MatchID', 'EventOrigin_x', 'EventOrigin_y',
            'EventDestination_x', 'EventDestination_y',
            'OwnScore', 'OpponentScore'}
float_type = {'EventTime'}


def read_info(path):
    def converter(data):
        for key, value in data.items():
            if key in float_type:
                data[key] = float(value)
            elif key in int_type:
                try:
                    data[key] = int(value)
                except ValueError:
                    pass
        return data

    with open(path) as f:
        reader = csv.DictReader(f)
        events = [converter(data) for data in reader]
    return events


def get_play_actions(events, filter_len=1):
    """
    Given a list of events occuring during a game, it splits the events
    into play actions using the following principle:

    - an action begins when a team gains ball possession
    - an action ends if one of three cases occurs:
    -- there is interruption of the match, due to: 1) end of first half or match; 2) ball
    out of the field 3) offside 4) foul

    """

    def is_save_attempt(event):
        return event['EventType'] == 'Save attempt'

    def is_duel(event):
        return event['EventType'] == 'Duel'

    def is_shot(event):
        return event['EventType'] == 'Shot' or event['EventType'] == 'Free Kick'

    def is_ball_lost(event, previous_event):
        if event['TeamID'] != previous_event['TeamID'] and not is_duel(event):
            return True

    def is_interruption(event, current_half):
        inter_type = {'Interruption', 'Foul', 'Offside'}
        event_type, match_period = event['EventType'], event['MatchPeriod']
        if event_type in inter_type or match_period != current_half:
            return True
        return False

    def add_actions(action):
        if len(action) >= filter_len:
            play_actions.append(action)

    play_actions = []

    current_action, current_half = [], '1H'
    previous_event = None
    for index, current_event in enumerate(events):
        current_event = events[index]

        # if the action stops by an game interruption
        if is_interruption(current_event, current_half):
            current_action.append(current_event)
            add_actions(current_action)
            current_action = []

        elif is_shot(current_event):
            # last event
            if index + 1 == len(events):
                current_action.append(current_event)
                add_actions(current_action)
                break

            next_event = events[index + 1]
            if is_interruption(next_event, current_half):
                current_action.append(current_event)
                current_action.append(next_event)
                add_actions(current_action)
                current_action = []

            ## IF THERE IS A SAVE ATTEMPT OR REFLEXES; GO TOGETHER
            elif is_save_attempt(next_event):
                current_action.append(current_event)
                current_action.append(next_event)
                add_actions(current_action)
                current_action = []

            else:
                current_action.append(current_event)
                add_actions(current_action)
                current_action = []

        elif previous_event is not None and is_ball_lost(current_event, previous_event):
            current_action.append(current_event)
            add_actions(current_action)
            current_action = [current_event]

        else:
            current_action.append(current_event)

        current_half = current_event['MatchPeriod']

        if not is_duel(current_event):
            previous_event = current_event

    return play_actions


def get_invasion_index(events_match, reduce=True):
    """
    Compute the invasion index for the input match
    """
    team2invasion_index, team2invasion_speed = ([[], []], [[], []])
    actions = get_play_actions(events_match, 1)
    off = max([x['EventTime'] for x in events_match if x['MatchPeriod'] == '1H'])
    for events in actions:
        offset = off if events[0]['MatchPeriod'] == '2H' else 0
        team_id = 0 if events[0]['TeamID'].startswith('H') else 1
        all_weights, times = [], []
        for ev in events:
            try:
                x, y, s = int(ev['EventOrigin_x']), int(ev['EventOrigin_y']), ev['EventTime']
            except ValueError:
                continue  # skip to next event in case of missing position data
            all_weights.append(get_weight((x, y)))
            # all_weights.append(get_datadriven_weight((x, y)))
            times.append(s)

        if len(all_weights) == 0:
            continue
        times_maxinv = times[int(np.argmax(np.array(all_weights)))]
        # times_maxinv = sorted(times, key=lambda x: all_weights[times.index(x)], reverse=True)[0]
        seconds = times_maxinv - events[0]['EventTime']
        if seconds > 0.8:
            if reduce:
                team2invasion_speed[team_id].append((np.max(all_weights) - all_weights[0]) / seconds ** 2)
            else:
                team2invasion_speed[team_id].append((events[0]['EventTime'] + offset,
                                                     (np.max(all_weights) - all_weights[0]) / seconds ** 2))
        if reduce:
            team2invasion_index[team_id].append(np.max(all_weights))
        else:
            team2invasion_index[team_id].append((events[0]['EventTime'] + offset, np.max(all_weights)))

    if reduce:
        for r in (team2invasion_index, team2invasion_speed):
            r[0] = sum(r[0]) / len(r[0])
            r[1] = sum(r[1]) / len(r[1])
    return team2invasion_index, team2invasion_speed


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
