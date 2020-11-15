import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx


def pitch():
    """
    code to plot a soccer pitch
    """
    # create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Pitch Outline & Centre Line
    plt.plot([0, 0], [0, 100], color="black")
    plt.plot([0, 100], [100, 100], color="black")
    plt.plot([100, 100], [100, 0], color="black")
    plt.plot([100, 0], [0, 0], color="black")
    plt.plot([50, 50], [0, 100], color="black")

    # Left Penalty Area
    plt.plot([16.5, 16.5], [80, 20], color="black")
    plt.plot([0, 16.5], [80, 80], color="black")
    plt.plot([16.5, 0], [20, 20], color="black")

    # Right Penalty Area
    plt.plot([83.5, 100], [80, 80], color="black")
    plt.plot([83.5, 83.5], [80, 20], color="black")
    plt.plot([83.5, 100], [20, 20], color="black")

    # Left 6-yard Box
    plt.plot([0, 5.5], [65, 65], color="black")
    plt.plot([5.5, 5.5], [65, 35], color="black")
    plt.plot([5.5, 0.5], [35, 35], color="black")

    # Right 6-yard Box
    plt.plot([100, 94.5], [65, 65], color="black")
    plt.plot([94.5, 94.5], [65, 35], color="black")
    plt.plot([94.5, 100], [35, 35], color="black")

    # Prepare Circles
    centreCircle = Ellipse((50, 50), width=30, height=39, edgecolor="black", facecolor="None", lw=1.8)
    centreSpot = Ellipse((50, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    leftPenSpot = Ellipse((11, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    rightPenSpot = Ellipse((89, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    # limit axis
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    ax.annotate("", xy=(25, 5), xytext=(5, 5),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.text(7, 7, 'Attack', fontsize=20)
    return fig, ax


def draw_pitch(pitch=None, line='black', view='full'):
    """
    Draw a soccer pitch given the pitch, the orientation, the view and the line

    Parameters
    ----------
    pitch

    """
    if view.lower().startswith("h"):
        fig, ax = plt.subplots(figsize=(6.8, 10.4))
        plt.xlim(49, 105)
        plt.ylim(-1, 69)
    else:
        fig, ax = plt.subplots(figsize=(10.4, 6.8))
        plt.xlim(-1, 105)
        plt.ylim(-1, 69)
    ax.axis('off')  # this hides the x and y ticks

    # side and goal lines #
    ly1 = [0, 0, 68, 68, 0]
    lx1 = [0, 104, 104, 0, 0]

    plt.plot(lx1, ly1, color=line, zorder=5)

    # boxes, 6 yard box and goals

    # outer boxes#
    ly2 = [13.84, 13.84, 54.16, 54.16]
    lx2 = [104, 87.5, 87.5, 104]
    plt.plot(lx2, ly2, color=line, zorder=5)

    ly3 = [13.84, 13.84, 54.16, 54.16]
    lx3 = [0, 16.5, 16.5, 0]
    plt.plot(lx3, ly3, color=line, zorder=5)

    # goals#
    ly4 = [30.34, 30.34, 37.66, 37.66]
    lx4 = [104, 104.2, 104.2, 104]
    plt.plot(lx4, ly4, color=line, zorder=5)

    ly5 = [30.34, 30.34, 37.66, 37.66]
    lx5 = [0, -0.2, -0.2, 0]
    plt.plot(lx5, ly5, color=line, zorder=5)

    # 6 yard boxes#
    ly6 = [24.84, 24.84, 43.16, 43.16]
    lx6 = [104, 99.5, 99.5, 104]
    plt.plot(lx6, ly6, color=line, zorder=5)

    ly7 = [24.84, 24.84, 43.16, 43.16]
    lx7 = [0, 4.5, 4.5, 0]
    plt.plot(lx7, ly7, color=line, zorder=5)

    # Halfway line, penalty spots, and kickoff spot
    ly8 = [0, 68]
    lx8 = [52, 52]
    plt.plot(lx8, ly8, color=line, zorder=5)

    plt.scatter(93, 34, color=line, zorder=5)
    plt.scatter(11, 34, color=line, zorder=5)
    plt.scatter(52, 34, color=line, zorder=5)

    alpha = 0 if pitch is None else 1
    circle1 = plt.Circle((93.5, 34), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=1, alpha=alpha)
    circle2 = plt.Circle((10.5, 34), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=1, alpha=alpha)
    circle3 = plt.Circle((52, 34), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=2, alpha=1)

    ## Rectangles in boxes
    rec1 = plt.Rectangle((87.5, 20), 16, 30, ls='-', color=pitch, zorder=1, alpha=alpha)
    rec2 = plt.Rectangle((0, 20), 16.5, 30, ls='-', color=pitch, zorder=1, alpha=alpha)

    ## Pitch rectangle
    rec3 = plt.Rectangle((-1, -1), 106, 70, ls='-', color=pitch, zorder=1, alpha=alpha)

    ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)

    return fig, ax


def draw_arrow(evs, ax, color):
    if evs[0]['EventOrigin_x'] != '':
        x, y = evs[0]['EventOrigin_x'] * 1.04, evs[0]['EventOrigin_y'] * 0.68
        ax.scatter(x, y, s=50, color=color, zorder=15)

    for ev in evs:
        if ev['EventOrigin_x'] == '':
            continue
        x, y = ev['EventOrigin_x'] * 1.04, ev['EventOrigin_y'] * 0.68
        if ev['EventDestination_x'] == '':
            ax.scatter(x, y, s=50, color=color, zorder=15)
            continue
        x2, y2 = ev['EventDestination_x'] * 1.04, ev['EventDestination_y'] * 0.68
        dx, dy = x2 - x, y2 - y
        ax.arrow(x, y, dx, dy, head_width=1, color=color)


def plot_passing(events, match_id):
    Huskies = [ev for ev in events if ev['TeamID'] == 'Huskies']
    Oppo = [ev for ev in events if ev['TeamID'] != 'Huskies']
    fig, ax = draw_pitch("#195905", "#faf0e6")

    draw_arrow(Huskies, ax, 'c')
    draw_arrow(Oppo, ax, 'k')
    plt.title(f'MatchID {match_id}', fontsize=20)
    plt.show()


def plot_kde_events_on_field(events, sample_size=10000):
    """
    Generate density plots on the field for each event type

    Parameters
    ----------
    sample_size: int
        random sample of values to use (default: 10000). The code becomes slow is you increase this value
        significantly.
    """
    position_ev = []
    event_type = {'Free Kick', 'Duel', 'Pass', 'Others on the ball',
                  'Foul', 'Offside', 'Shot'}
    for ev in events:
        if ev['EventType'] in event_type:
            position_ev.append([ev['EventType'], ev['EventOrigin_x'] * 1.04, ev['EventOrigin_y'] * 0.68])

    df_pos_ev = pd.DataFrame(position_ev, columns=['EventType', 'x', 'y'])

    for event in np.unique(df_pos_ev['EventType']):
        print(event)
        df_pos_event = df_pos_ev[df_pos_ev['EventType'] == event]
        fig, ax = draw_pitch()
        if len(df_pos_event) >= 10000:
            x_y = df_pos_event[['x', 'y']].sample(sample_size).astype(float)
        else:
            x_y = df_pos_event[['x', 'y']].astype(float)
        sns.kdeplot(x_y['x'], x_y['y'], cmap='Greens', shade=True)
        plt.title(event, fontsize=20)
        plt.xlim(-1, 105)
        plt.ylim(-1, 69)
        plt.axis('off')
        fig.tight_layout()
        plt.show()


def passing_network(events):
    g1 = nx.Graph()
    g2 = nx.Graph()
    for ev in events:
        p1, p2 = ev['OriginPlayerID'], ev['DestinationPlayerID']
        g = g1 if p1.startswith('H') else g2
        p1, p2 = p1[p1.rfind('_') + 1:], p2[p2.rfind('_') + 1:]
        if g.has_edge(p1, p2):
            g[p1][p2]['weight'] += 1
        else:
            g.add_edge(p1, p2, weight=1)

    def draw_graph(g):
        weight = [g[u][v]['weight'] / 5 for u, v in g.edges()]
        # pos = nx.kamada_kawai_layout(g)
        pos = nx.spring_layout(g)
        # plt.subplot(111)
        # nx.draw(g, pos, with_labels=True, width=weight)
        # plt.show()
        connectivity = nx.algebraic_connectivity(g)
        centrality = nx.global_reaching_centrality(g)
        # print(f'density = {nx.density(g):.4f}')
        return connectivity, centrality

    H_conn, H_cent = draw_graph(g1)
    O_conn, O_cent = draw_graph(g2)
    return (H_conn, H_cent), (O_conn, O_cent)
