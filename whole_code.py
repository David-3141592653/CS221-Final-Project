import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import subprocess # for calling the oracle

device = "cpu"
#print(device)

# ==========================================================
# ENVIRONMENT
# ==========================================================

class Connect4Env:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.move_history = [] # exists exclusively for running the oracle
        return self.get_state()

    # no perspective flipping
    def get_state(self):
        return np.stack([
            (self.board == self.current_player).astype(np.float32),
            (self.board == -self.current_player).astype(np.float32)
        ])

    def legal_actions(self):
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def legal_mask_array(self):
        mask = np.zeros(self.COLS, dtype=np.float32)
        mask[self.legal_actions()] = 1.0
        return mask

    def legal_mask_no_normalize(self, pi):
        legal = self.legal_actions()
        masked = np.zeros_like(pi)
        masked[legal] = pi[legal]

        return masked

    def normalize(self, masked, legal):
        s = np.sum(masked)
        if s > 0:
            masked = masked / s
        else:
            masked[legal] = 1.0 / len(legal)

        return masked

    def legal_mask(self, pi):
        return self.normalize(self.legal_mask_no_normalize(pi), self.legal_actions())

    def step(self, action):

        if action not in self.legal_actions():
            raise ValueError("Illegal move")

        # place piece
        for r in reversed(range(self.ROWS)):
            if self.board[r, action] == 0:
                self.board[r, action] = self.current_player
                break

        self.current_player *= -1

        # again, update this solely for the oracle
        self.move_history.append(action)

        if self.finished():
            return self.get_state(), True
        else:
            return self.get_state(), False

    def check_win(self, player):

        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r, c+i] == player for i in range(4)):
                    return True

        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if all(self.board[r+i, c] == player for i in range(4)):
                    return True

        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True

        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True

        return False

    def winner(self):
        if self.check_win(1):
            return 1
        elif self.check_win(-1):
            return -1
        elif len(self.legal_actions()) == 0:
            return 0

        return None

    def finished(self):
        return self.winner() is not None

    def render(self):
        for row in self.board:
            line = []

            for cell in row:
                if cell == 1:
                    line.append(f"🔴")      # Player 1
                elif cell == -1:
                    line.append(f"🟡")   # Player 2
                else:
                    line.append(f"🔵")     # Empty slot

            print(" ".join(line))

        print()

  # ==========================================================
# REPLAY BUFFER
# ==========================================================

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def push(self, state, pi, reward, legal_mask):
        self.buffer.append((state, pi, reward, legal_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, pi, reward, legal_mask = zip(*batch)

        return (
            np.array(state),
            np.array(pi),
            np.array(reward, dtype=np.float32),
            np.array(legal_mask, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ==========================================================
# NETS
# ==========================================================

# Recurrent Convolutional Towers
class TowerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        return self.relu(x + residual)

# Full Connect 4 neural net
class AlphaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Conv2d(2, 64, kernel_size=3, padding=1)

        self.tower = nn.Sequential(
            TowerBlock(64),
            TowerBlock(64),
            TowerBlock(64),
            TowerBlock(64),
        )

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7)
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

# ==================================================
# SAVING AND LOADING
# ==================================================

import time
import torch

# for saving directly into google drive
from google.colab import drive
drive.mount('/content/drive')

def save_net(net, optimizer, epoch):
    filename = f"/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_{epoch}.pt"

    torch.save({
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, filename)

    print(f"Saved: {filename}")


def load_net(net, optimizer, net_filepath):
    checkpoint = torch.load(net_filepath, map_location=device)

    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"]

    print(f"Loaded {net_filepath}")
    print(f"Checkpoint epoch: {epoch}")

    return epoch

def load_net_for_play(net, net_filepath):
    checkpoint = torch.load(net_filepath, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

#save_net(net, optimizer, 360)

# ==========================================================
# Monte-Carlo Tree Search
# ==========================================================

class Node:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}   # action -> Node

        self.N = 0
        self.W = 0.0         # total value
        self.P = prior

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

class MCTS:
    def __init__(self, net, env, c_puct=1.5, sims=200, debug=False, eval=False):
        self.net = net
        self.c_puct = c_puct
        #print("c_puct type:", type(c_puct))

        self.sims = sims

        self.debug = debug

        self.eval = eval

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    def search(self, env):
        self.root = Node()

        # expand root once
        policy, value = self._evaluate(env)
        self._expand(self.root, env, policy)

        for _ in range(self.sims):
            self._simulate(env)

        if self.debug:
            for a, child in self.root.children.items():
                print((a, "--> N:", child.N, "W:", child.W, "P:", child.P, "Q:", child.Q()))

        if self.eval:
            print("evaluation:", self.root.Q())

        return self._get_policy(self.root)

    # --------------------------------------------------
    # SINGLE SIMULATION
    # --------------------------------------------------
    def _simulate(self, env):
        node = self.root
        sim_env = self._clone(env)
        path = []

        if self.debug:
            print("ROOT ID:", id(self.root))

        while True:

            # ---------------------------
            # TERMINAL CHECK
            # ---------------------------
            if self._terminal(sim_env):

                if self.debug:
                    print("Terminal value reached")

                path.append((node, None))

                value = self._value(sim_env)
                self._backprop(path, value)
                return

            # ---------------------------
            # EXPAND LEAF
            # ---------------------------
            if len(node.children) == 0:
                policy, value = self._evaluate(sim_env)

                if node.N == 0:
                    self._expand(node, sim_env, policy)

                self._backprop(path, value)
                return

            # and now ... puct!!!
            assert len(node.children) > 0 or self._terminal(sim_env)
            action, next_node = self._select(node)
            assert next_node is node.children[action]

            if self.debug:
                print("simulation action:", action)

            sim_env.step(action)
            path.append((node, action))

            if self.debug:
                print("TREE PATH:", path)

            if self.debug:
                print("path length thus far:", len(path))

            node = next_node

    # --------------------------------------------------
    # SELECTION (PUCT)
    # --------------------------------------------------
    def _select(self, node):
        best_score = -1e9
        best_action = None
        best_child = None

        sqrt_N = np.sqrt(node.N + 1e-8)

        for action, child in node.children.items():
          
            u = self.c_puct * child.P * sqrt_N / (1 + child.N)
            q = child.Q()

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    # --------------------------------------------------
    # EXPANSION
    # --------------------------------------------------
    def _expand(self, node, env, policy):
        legal = env.legal_actions()

        for a in legal:
            node.children[a] = Node(
                parent=node,
                prior=policy[a]
            )
            if self.debug:
                print("EXPANDING:", a, id(node), "->", id(node.children[a]))

    # --------------------------------------------------
    # BACKPROP
    # --------------------------------------------------
    def _backprop(self, path, value):
        if self.debug:
            print("BACKPROP ROOT ID:", id(self.root))

            print("BACKPROP START VALUE:", value)
            for node, action in reversed(path):
                print("  node update:", value)

            # check for which actions have been picked
            if len(path) > 0:
                root_node, root_action = path[0]
                if root_node is self.root:
                    print("ROOT ACTION BACKPROP:", root_action)

            print("ROOT CHILD IDS:")
            for a, c in self.root.children.items():
                print(a, id(c), c.N)

        for node, action in reversed(path):

            if self.debug:
                print("UPDATING NODE:", action, id(node))

            node.N += 1
            node.W += value
            value = -value

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------
    def _evaluate(self, env):
        state = torch.tensor(
            env.get_state(),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = self.net(state)

        #print("POLICY LOGITS TYPE:", type(policy_logits))
        #print("VALUE TYPE:", type(value))

        policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        value = value.item()

        # mask illegal moves
        legal = env.legal_actions()
        masked = np.zeros_like(policy)
        masked[legal] = policy[legal]

        s = masked.sum()
        if s > 0:
            masked /= s
        else:
            masked[legal] = 1.0 / len(legal)

        #print("MASKED TYPE:", type(policy_logits))
        #print("VALUE TYPE:", type(value))

        return masked, value

    # --------------------------------------------------
    # TERMINAL
    # --------------------------------------------------
    def _terminal(self, env):
        return env.winner() is not None

    def _value(self, env):
        # value from current player's perspective
        return env.winner() * env.current_player * -1

    # --------------------------------------------------
    # CLONE
    # --------------------------------------------------
    def _clone(self, env):
        new_env = type(env)()
        new_env.board = env.board.copy()
        new_env.current_player = env.current_player
        return new_env

    # --------------------------------------------------
    # FINAL POLICY
    # --------------------------------------------------
    def _get_policy(self, root):
        visits = np.zeros(7)

        for a, child in root.children.items():
            visits[a] = child.N

        if visits.sum() == 0:
            legal = list(root.children.keys())
            if len(legal) == 0:
                return np.ones(7) / 7
            policy = np.zeros(7)
            policy[legal] = 1.0 / len(legal)
            return policy

        return visits / visits.sum()

  # ==================================================
# MCTS TESTING
# ==================================================

# winning move miss issue:
def check_winning_move_miss(env, net_filepath):
    env = Connect4Env()

    # set up board
    env.board = env.board = np.zeros((env.ROWS, env.COLS), dtype=np.int8)

    env.step(3)  # X
    env.step(3)  # O

    env.step(0)  # X
    env.step(2)  # O

    env.step(3)  # X
    env.step(4)  # O

    env.step(2)  # X
    env.step(4)  # O

    env.step(2)  # X
    env.step(4)  # O

    env.step(4)  # X
    env.step(5)  # O

    print("legal:", env.legal_actions())

    # load network
    net = AlphaNet().to(device)
    checkpoint = torch.load(net_filepath, map_location=device)

    net.load_state_dict(checkpoint["state_dict"])

    mcts = MCTS(net, env, sims=100, debug=True)


    # choose move
    pi = mcts.search(env)
    print(pi)
    move = int(np.argmax(env.legal_mask(pi)))

    print(f"AI plays: {move}")
    env.step(move)
    env.render()

def check_losing_move_miss(env):
    pass

#check_winning_move_miss(Connect4Env(), "/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_460.pt")

# ==================================================
# SELF-PLAY
# ==================================================

def self_play(env, mcts, replay_buffer):
    state = env.reset()
    done = False
    game_history = []

    move_idx = 0

    while not done:

        # get policy
        pi = mcts.search(env)

        # get temperature
        temp = get_temperature(move_idx)

        # get action
        action = pick_action(env, pi, temp)

        # update game history
        game_history.append((state.copy(), pi.copy(), env.current_player, env.legal_mask_array()))

        # next position
        state, done = env.step(action)

        move_idx += 1

    # now, update rewards based on who won
    winner = env.winner()

    for state, pi, player, legal_mask in game_history:

        # reward 0 for draw, 1 if you are the winner, -1 if you are the loser -
        # little trick here is just multiply
        reward = winner * player

        replay_buffer.push(state, pi, reward, legal_mask)

    return winner

# helper method to get temperature (TODO: MESS AROUND WITH THIS)
def get_temperature(move_idx):
    # classic AlphaZero-style schedule
    if move_idx < 10:
        return 1.0
    else:
        return 0.1

# helper method to get action from policy
def pick_action(env, pi, temp):
    pi = np.array(pi)

    masked = env.legal_mask(pi)

    if temp == 0:
        return np.argmax(masked)

    if temp != 1:
        masked = np.exp(np.log(masked + 1e-8) / temp) # TWEAK +1e-8?

    masked /= np.sum(masked)
    return np.random.choice(np.arange(7), p=masked)

# ==================================================
# TRAINING
# ==================================================
def learn_from_game_batch(net, env, optimizer, buffer, batch_size, policy_to_value_loss_ratio=1, debug=False):

    # get training sample
    states, pis, rewards, legal_masks = buffer.sample(batch_size)

    if 1 == 0: #debug:
        print("Batch stats:")
        print("states:", states.shape)
        print("pis mean:", np.mean(pis))
        print("rewards mean:", np.mean(rewards))

    # convert to torch tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    pis = torch.tensor(pis, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    legal_masks = torch.tensor(legal_masks).to(device)

    if 1 == 0: #debug:
        print("illegal pi mass:", (pis * (1 - legal_masks)).sum(dim=1)[:5])

    # predictions
    pred_policy, pred_value = net(states)

    if 1 == 0: #debug:
        print("\nPOLICY MASKING CHECK")

        print("mask sum per sample (first 5):",
              legal_masks.sum(dim=1)[:5].detach().cpu().numpy())

        print("pred_policy min/max:",
              pred_policy.min().item(),
              pred_policy.max().item())

        print("any NaNs in logits:",
              torch.isnan(pred_policy).any().item())

        print("pis sum:", pis.sum(dim=1)[:5])
        print("masked legal sum:", legal_masks.sum(dim=1)[:5])
        print("pis max per sample:", pis.max(dim=1)[0][:5])

    if 1 == 0: #debug:
        print("pred_policy mean/std:",
            pred_policy.mean().item(),
            pred_policy.std().item())
        print("pred_value mean:",
            pred_value.mean().item())

    # loss
    log_probs = F.log_softmax(pred_policy, dim=1)
    policy_loss = -(pis * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_value, rewards)

    n = policy_to_value_loss_ratio
    loss = policy_loss * (n / (n+1)) + value_loss * (1/(n+1))

    if 1 == 0: #debug:
        print("policy loss:", policy_loss.item())
        print("value loss:", value_loss.item())
        print("total loss:", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if 1 == 0: #debug:
        total_norm = sum(p.data.norm().item() for p in net.parameters())
        print("weight norm:", total_norm)

    return loss

def train(net, env, optimizer, mcts, buffer, batch_size=100, start_epoch=0, epochs=1000, self_plays_per_epoch=100, batch_trains_per_epoch=100, debug=False):
    # update sims number based on training resuming
    for e in range(start_epoch // 100 + 1):
        mcts.sims += min(max(700-mcts.sims, 0), int(100 / (e+100) * 100))

    for e in range(start_epoch, epochs + start_epoch):
        print("epoch:", e+1)

        # play
        for p in range(self_plays_per_epoch):
            self_play(env, mcts, buffer)

        #print("buffer length:", len(buffer))

        # make sure we get a sizeable buffer before training begins
        if len(buffer) < batch_size:
            continue

        # training step
        for t in range(batch_trains_per_epoch):
            learn_from_game_batch(net, env, optimizer, buffer, batch_size, debug=debug)

        # printouts
        if e % 10 == 0:
            print("epoch, len(buffer):", e+1, len(buffer))
            if debug:
                pass #learning_check(net, mcts, env)
            if e % 10 == 0:
                save_net(net, optimizer, e)
                if e % 100 == 0:
                    mcts.sims += min(max(700-mcts.sims, 0), int(100 / (e+100) * 100))

                    print("updated sims:", mcts.sims)

# ==================================================
# TESTING
# ==================================================

# test that MCTS runs the way one expects
def run_self_play_test(env, mcts, num_games=20, verbose=True):
    results = {1: 0, -1: 0, 0: 0}

    for game in range(num_games):

        state = env.reset()
        done = False
        move_count = 0

        while not done:

            # get MCTS policy
            pi = mcts.search(env)

            # deterministic play (no temperature)
            action = np.argmax(env.legal_mask(pi))

            state, done = env.step(action)
            move_count += 1

        winner = env.winner()
        results[winner] += 1

        if verbose:
            if winner == 1:
                print(f"Game {game+1}: Player 1 wins in {move_count} moves")
            elif winner == -1:
                print(f"Game {game+1}: Player -1 wins in {move_count} moves")
            else:
                print(f"Game {game+1}: Draw in {move_count} moves")

    print("\n=== SUMMARY ===")
    print(f"Player 1 wins: {results[1]}")
    print(f"Player -1 wins: {results[-1]}")
    print(f"Draws: {results[0]}")

# training tests

# this allows us to investigate the weights (good luck with that ...)
def clone_state_dict(net):
    return {k: v.clone() for k, v in net.state_dict().items()}

# check change after one training batch
def one_batch_change(net, optimizer, buffer):
    before = clone_state_dict(net)
    learn_from_game_batch(net, env, optimizer, buffer, batch_size=32)
    after = clone_state_dict(net)

    changed = False

    for k in before:
        if not torch.equal(before[k], after[k]):
            print("Changed:", k)
            changed = True

    if not changed:
        print("WARNING: No learning happening (weights unchanged)")
    else:
        print("Learning is happening! (however good it may be)")

def training_loop_integration_test(net, env, optimizer, mcts, buffer, start_epoch = 0):

    print("\nRUNNING FULL TRAIN PIPELINE TEST\n")

    # run a tiny training cycle
    train (
        net=net,
        env=env,
        optimizer=optimizer,
        mcts=mcts,
        buffer=buffer,
        start_epoch=start_epoch,
        epochs=1000,
        self_plays_per_epoch=10,
        batch_trains_per_epoch=10,
        debug=True
    )

    print("\nPIPELINE TEST COMPLETE\n")

# --------------------------------------------------
# GRANDMASTER TESTS - PLAY LEVEL IMPROVEMENT?
# --------------------------------------------------

"""
We will check the following as heuristics to whether learning is truly happening

1. mcts integration into learning
--> we want to see that the mcts is affecting the nets decisions by making it conform more to mcts policy - checks pi grad. desc.
--> to do so, count agreements between mcts and net action decisions - the higher the better

2. mcts vs net
--> throughout "middle" of training we want the mcts to beat the net - after all, it's a neural net with exploration ON TOP of it
--> we play mcts vs net and see if mcts is better
--> if mcts plays better (this test) and mcts is integrated into net policy (test 1), then that's an immediate indication of growth
"""

def mcts_integration(net, mcts, env, n=20):
    agreements = 0
    total_moves = 0

    for _ in range(n):
        state = env.reset()

        done = False
        while not done:
            pi_mcts = mcts.search(env)

            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = net(state_t)
                pi_net = torch.softmax(logits, dim=1).cpu().numpy()[0]

            action_net = np.argmax(pi_net)
            action_mcts = np.argmax(pi_mcts)

            if action_net == action_mcts:
                agreements += 1

            total_moves += 1

            state, done = env.step(action_mcts)

    return agreements / total_moves

def net_vs_mcts(net, mcts, env, games=50):
    net_wins = 0
    mcts_wins = 0
    draws = 0
    num_games = 0
    total_moves = 0

    for g in range(games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == 1:
                # MCTS player
                pi = mcts.search(env)
                action = np.argmax(pi)
            else:
                # raw net player
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = net(state)

                legal = env.legal_actions()
                mask = torch.full_like(logits, -1e9)
                mask[:, legal] = 0

                masked_logits = logits + mask

                pi = torch.softmax(masked_logits, dim=1)
                action = torch.argmax(pi, dim=1).item()

            legal = env.legal_actions()

            #player_str = "MCTS" if env.current_player == 1 else "NET"
            #print("Player:", player_str)
            #print("Chosen:", action)
            #print("Legal:", legal)
            #print("Root policy:", pi)

            assert action in legal, f"{player_str} produced illegal move"

            state, done = env.step(action)

            total_moves += 1

        w = env.winner()
        if w == 1:
            mcts_wins += 1
        elif w == -1:
            net_wins += 1
        else:
            draws += 1

        num_games += 1

    return net_wins, mcts_wins, draws, total_moves / num_games

def learning_check(net, mcts, env):
    print("\n===== AND NOW FOR SOMETHING COMPLETELY DIFFERENT ... a short intermission. =====")

    # agreements
    agreement = mcts_integration(net, mcts, env, n=5)
    print(f"MCTS-Net move agreement: {agreement:.3f}")

    # mcts vs net head-on encounters (IN THE WILD!!!)
    net_w, mcts_w, draws, avg_game_length = net_vs_mcts(net, mcts, env, games=20)
    print(f"Net vs MCTS: {net_w}W / {mcts_w}L / {draws}D")
    print("Average game length:", avg_game_length)

    print("================================\n")

# ==================================================
# PROGRESS CHECK TEST: AI VS. YOUNGER VERSION
# ==================================================

def play_vs_younger(wise_mcts, silly_mcts, env, wise_first, verbose, render):
    state = env.reset()
    done = False

    while not done:

        if env.current_player == (1 if wise_first else -1):
            # WISE TURN
            pi = wise_mcts.search(env)
            #move = np.random.choice(np.arange(7), p=env.legal_mask(pi)) # play with some randomness to get more statistically representative result
            move = np.argmax(env.legal_mask(pi))

            if render:
                print(f"Wise Net plays: {move}")

        else:
            # SILLY TURN (MCTS)
            pi = silly_mcts.search(env)
            #move = np.random.choice(np.arange(7), p=env.legal_mask(pi))
            move = np.argmax(env.legal_mask(pi))

            if render:
                print(f"Silly Net plays: {move}")

        state, done = env.step(move)
        if render:
            env.render()

    winner = env.winner()

    if winner == (1 if wise_first else -1):
        if verbose:
            print("Good game, my young up and coming friend.")
        return (1,0,0) # boolean triple representing wise win, silly win, or draw, in that order
    elif winner != 0:
        if verbose:
            print("WAHOOO TAKE THAT OLD MAN ... YEAAAHHH!!")
        return (0,1,0)
    else:
        if verbose:
            print("Draw")
        return (0,0,1)

def test_progress(wise_mcts, silly_mcts, env, verbose, num_games, render):
    wise_wins = 0
    silly_wins = 0
    draws = 0

    for i in range(num_games):
        w, s, d = play_vs_younger(wise_mcts, silly_mcts, env, wise_first= (True if i % 2 == 0 else False), verbose=verbose, render=render)
        wise_wins += w
        silly_wins += s
        draws += d

    print(f"Wise wins: {wise_wins}")
    print(f"Silly wins: {silly_wins}")
    print(f"Draws: {draws}")

def test_progress_of_epochs(wise_epoch, silly_epoch, verbose=False, num_games=100, render=False):
    print(f"\nEPOCH {wise_epoch} vs {silly_epoch}:\n")

    env = Connect4Env()

    silly_net = AlphaNet().to(device)
    wise_net = AlphaNet().to(device)

    wise_checkpoint = torch.load(f"/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_{wise_epoch}.pt", map_location=device)
    silly_checkpoint = torch.load(f"/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_{silly_epoch}.pt", map_location=device)

    silly_net.load_state_dict(silly_checkpoint["state_dict"])
    wise_net.load_state_dict(wise_checkpoint["state_dict"])

    silly_mcts = MCTS(silly_net, env, sims=500)
    wise_mcts = MCTS(wise_net, env, sims=500)

    test_progress(wise_mcts, silly_mcts, env, verbose, num_games, render)


#env = Connect4Env()

test_progress_of_epochs(470, 250, num_games=1, render=True)
#test_progress_of_epochs(350, 250)
#test_progress_of_epochs(350, 200)
#test_progress_of_epochs(350, 150)
#test_progress_of_epochs(350, 100)
#test_progress_of_epochs(350, 51)

"""
# ==================================================
# WHAT I RUN TO INITIATE TRAINING
# ==================================================

env = Connect4Env()
net = AlphaNet().to(device)
mcts = MCTS(net, env, c_puct=1.5, sims=100)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
buffer = ReplayBuffer()

#training_loop_integration_test(net, env, optimizer, mcts, buffer)
"""

"""
# ==================================================
# WHAT I RUN TO RESUME TRAINING
# ==================================================
def resume_training(net_filepath):
    env = Connect4Env()
    net = AlphaNet().to(device)
    mcts = MCTS(net, env, sims=200)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer()

    start_epoch = load_net(net, optimizer, net_filepath)

    training_loop_integration_test(net, env, optimizer, mcts, buffer, start_epoch)

resume_training("/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_500.pt")
"""

# ==================================================
# PLAY VS. HUMAN
# ==================================================

def play_vs_ai(net, mcts, env, human_first=True, eval=False):
    state = env.reset()
    done = False

    print("You are X (1) and AI is O (-1)" if human_first else
          "AI is X (1) and you are O (-1)")

    while not done:
        env.render()

        if env.current_player == (1 if human_first else -1):
            # HUMAN TURN
            legal = env.legal_actions()
            print("Legal moves:", legal)

            move = int(input("Your move (0-6): "))

            if move not in legal:
                print("Illegal move, try again.")
                continue

        else:
            # AI TURN (MCTS)
            pi = mcts.search(env)

            move = int(np.argmax(env.legal_mask(pi)))
            print(f"AlphaConnect plays: {move}")

        state, done = env.step(move)

    env.render()

    winner = env.winner()

    if winner == 1:
        print("Player 1 wins")
    elif winner == -1:
        print("Player -1 wins")
    else:
        print("Draw")
      
"""
net = AlphaNet().to(device)
checkpoint = torch.load("/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_470.pt", map_location=device)

net.load_state_dict(checkpoint["state_dict"])

env = Connect4Env()
mcts = MCTS(net, env, sims=1000, eval=True)

play_vs_ai(net, mcts, env, human_first=False)
"""

# ==================================================
# RUN ORACLE ON STATE
# ==================================================

# returns a set of scores for each legal move
def oracle_analyze(sequence):
    # run solver to get score on all moves
    result = subprocess.run(
        ["./c4solver", "-a"],
        input=sequence + "\n",
        text=True,
        capture_output=True,
        cwd="/content/oracle"
    )

    scores = result.stdout.strip().split()[1:]
    return np.array([int(s) for s in scores])

"""
Analyzes action when applied to the env
ALWAYS RUN WITH env given BEFORE action applied

Returns
-- best_score - action_score
-- is_optimal
"""
def oracle_analyze_action(sequence, action):
    scores = oracle_analyze(sequence)
    return max(scores) - scores[action], max(scores) == scores[action]

# --------------------------------------------------
# SAME FUNCTIONS, BUT WORK WITH ENV
# --------------------------------------------------

def oracle_analyze_env(env):
    sequence = ''.join(str(col+1) for col in env.move_history)
    return oracle_analyze(sequence)

def oracle_analyze_env_action(env, action):
    scores = oracle_analyze_env(env)
    return max(scores) - scores[action], max(scores) == scores[action]

# ==================================================
# TEST THE MODEL AGAINST THE ORACLE
# ==================================================

"""
Metric 1:

Have model play itself starting from 10 completely uniformly at random positions
Analyze accuracy according to that

Picked first three moves as center, center, center
as the models have consistently learned those are the best moves -
variety only begins with the second player's second move
"""

def random_position(env, num_random_moves=0):
    env.reset()

    env.step(3)
    env.step(3)
    env.step(3)

    for _ in range(num_random_moves):
        legal_moves = np.flatnonzero(env.legal_mask_array())
        action = np.random.choice(legal_moves)
        _, done = env.step(action)
        if done:
            break
    return env.get_state()

# --------------------------------------------------
# EVALUATE ONE SELF-PLAY GAME
# --------------------------------------------------

def evaluate_self_play_game(env, mcts):

    done = False
    move_idx = 0

    score_differences = []
    optimal_moves = []

    # identical to self_play function, essentially
    while not done:

        # get policy
        pi = mcts.search(env)

        # get temperature
        temp = get_temperature(len(env.move_history)) # include starting pos moves in temp calculation

        # get action
        action = pick_action(env, pi, temp)

        # Evaluate BEFORE applying action - see oracle_analyze documentation
        difference, is_optimal = oracle_analyze_env_action(env, action)

        score_differences.append(difference)
        optimal_moves.append(is_optimal)

        # next position
        _, done = env.step(action)

        move_idx += 1

    return score_differences, optimal_moves

def evaluate_random_game(env):

    done = False
    move_idx = 0

    score_differences = []
    optimal_moves = []

    # identical to self_play function, essentially
    while not done:

        # get policy
        pi = np.array([1/7,1/7,1/7,1/7,1/7,1/7,1/7])

        # get action
        action = pick_action(env, pi, 1)

        # Evaluate BEFORE applying action - see oracle_analyze documentation
        difference, is_optimal = oracle_analyze_env_action(env, action)

        score_differences.append(difference)
        optimal_moves.append(is_optimal)

        # next position
        _, done = env.step(action)

        move_idx += 1

    return score_differences, optimal_moves

# ==================================================
# ORACLE SELF-PLAY EVALUATION
# ==================================================

def evaluate_model_oracle(env, mcts, num_positions=10):

    all_differences = []
    all_optimal = []

    for i in range(num_positions):

        # Generate random starting position
        random_position(env)

        # Play a complete self-play game
        differences, optimal = evaluate_self_play_game(env, mcts)

        all_differences.extend(differences)
        all_optimal.extend(optimal)

        print(
            f"Game {i+1}/{num_positions} | "
            f"Moves: {len(differences)} | "
            f"Avg difference: {np.mean(differences):.3f} | "
            f"Optimal: {100*np.mean(optimal):.1f}%"
        )

    # get averages
    avg_difference = np.mean(all_differences)
    optimal_percentage = 100 * np.mean(all_optimal)

    print("\n========== FINAL RESULTS ==========")
    print(f"Positions tested: {num_positions}")
    print(f"Total moves: {len(all_differences)}")
    print(f"Average optimal score difference: {avg_difference:.4f}")
    print(f"Percentage of optimal moves: {optimal_percentage:.2f}%")

    return avg_difference, optimal_percentage

def evaluate_random_baseline(env, num_positions=10):

    all_differences = []
    all_optimal = []

    for i in range(num_positions):

        # Generate random starting position
        random_position(env)

        # Play a complete self-play game
        differences, optimal = evaluate_random_game(env)

        all_differences.extend(differences)
        all_optimal.extend(optimal)

        print(
            f"Game {i+1}/{num_positions} | "
            f"Moves: {len(differences)} | "
            f"Avg difference: {np.mean(differences):.3f} | "
            f"Optimal: {100*np.mean(optimal):.1f}%"
        )

    # get averages
    avg_difference = np.mean(all_differences)
    optimal_percentage = 100 * np.mean(all_optimal)

    print("\n========== FINAL RESULTS ==========")
    print(f"Positions tested: {num_positions}")
    print(f"Total moves: {len(all_differences)}")
    print(f"Average optimal score difference: {avg_difference:.4f}")
    print(f"Percentage of optimal moves: {optimal_percentage:.2f}%")

    return avg_difference, optimal_percentage

def get_mcts(epoch):
    net = AlphaNet().to(device)
    checkpoint = torch.load(f"/content/drive/MyDrive/AlphaConnects/AlphaConnect_epoch_{epoch}.pt", map_location=device)

    net.load_state_dict(checkpoint["state_dict"])

    return MCTS(net, env, sims=500)

def evaluate_model_oracle_at_epoch(epoch, num_positions=10):
    return evaluate_model_oracle(Connect4Env(), get_mcts(epoch), num_positions)


"""
avg_diff, optimal_pct = evaluate_model_oracle_at_epoch(51)
"""

"""
avg_diff, optimal_pct = evaluate_random_baseline(Connect4Env())
"""

