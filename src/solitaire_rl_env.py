import random

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import gymnasium as gym
from gymnasium import spaces

import numpy as np

# -------------------------
# Card utilities
# -------------------------
SUITS = [0, 1, 2, 3]  # 0♠,1♥,2♣,3♦ (you can map however you like)
SUIT_NAMES = {0: "S", 1: "H", 2: "C", 3: "D"}
RANKS = list(range(1, 14))  # 1..13 (Ace..King)

NUM_TABLEAU = 7
MAX_TABLEAU_HEIGHT = 19  # worst-case upper bound (19 is sufficient for Klondike)

# Action layout sizes
N_DRAW = 1
N_WASTE_TO_FOUNDATION = 1
N_WASTE_TO_TABLEAU = NUM_TABLEAU
N_TAB_TO_FOUNDATION = NUM_TABLEAU
N_TAB_RUN_TO_TAB = NUM_TABLEAU * (NUM_TABLEAU - 1) * 13  # i!=j and depth 1..13

ACTION_OFFSET_DRAW = 0
ACTION_OFFSET_WASTE_TO_FOUNDATION = ACTION_OFFSET_DRAW + N_DRAW  # 1
ACTION_OFFSET_WASTE_TO_TABLEAU = ACTION_OFFSET_WASTE_TO_FOUNDATION + N_WASTE_TO_FOUNDATION  # 2
ACTION_OFFSET_TAB_TO_FOUNDATION = ACTION_OFFSET_WASTE_TO_TABLEAU + N_WASTE_TO_TABLEAU  # 9
ACTION_OFFSET_TAB_RUN_TO_TAB = ACTION_OFFSET_TAB_TO_FOUNDATION + N_TAB_TO_FOUNDATION  # 16

ACTION_SPACE_N = (
        N_DRAW
        + N_WASTE_TO_FOUNDATION
        + N_WASTE_TO_TABLEAU
        + N_TAB_TO_FOUNDATION
        + N_TAB_RUN_TO_TAB
)


def card_id(suit: int, rank: int) -> int:
    """0..51 encoding. rank in 1..13, suit in 0..3."""
    return suit * 13 + (rank - 1)


def id_to_card(cid: int) -> Tuple[int, int]:
    suit = cid // 13
    rank = (cid % 13) + 1
    return suit, rank


def is_red(suit: int) -> bool:
    return suit in (1, 3)  # hearts, diamonds


def is_black(suit: int) -> bool:
    return not is_red(suit)


@dataclass
class Pile:
    face_down: List[int]
    face_up: List[int]

    def copy(self) -> "Pile":
        return Pile(self.face_down.copy(), self.face_up.copy())


class KlondikeSolitaireEnv(gym.Env):
    metadata = {"render.modes": ["ansi"], "render_fps": 0}

    def __init__(self, seed: Optional[int] = None, step_limit: int = 1000):
        super().__init__()
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        else:
            self.np_random, seed = gym.utils.seeding.np_random()

        self.step_limit = step_limit

        # Observation space (Dict of padded arrays)
        self.observation_space = spaces.Dict(
            {
                # Tableau cards: padded with -1
                "tableau_cards": spaces.Box(
                    low=-1, high=51, shape=(NUM_TABLEAU, MAX_TABLEAU_HEIGHT), dtype=np.int16
                ),
                # Face-up mask for tableau (0/1)
                "tableau_faceup": spaces.Box(
                    low=0, high=1, shape=(NUM_TABLEAU, MAX_TABLEAU_HEIGHT), dtype=np.int8
                ),
                # Foundations: top rank per suit (0..13)
                "foundations": spaces.Box(low=0, high=13, shape=(4,), dtype=np.int8),
                # Waste top card id (-1 if none)
                "waste_top": spaces.Box(low=-1, high=51, shape=(1,), dtype=np.int16),
                # Stock count
                "stock_count": spaces.Box(low=0, high=52, shape=(1,), dtype=np.int16),
                # Steps used
                "steps": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.int32),
            }
        )

        self.action_space = spaces.Discrete(ACTION_SPACE_N)

        # Internal state containers
        self.tableau: List[Pile] = []
        self.foundations: List[int] = [0, 0, 0, 0]  # top rank per suit (0 means empty)
        self.stock: List[int] = []  # face-down stack (top at end)
        self.waste: List[int] = []  # face-up stack (top at end)
        self.steps = 0

    # --------------- Game Setup ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)

        deck = list(range(52))
        random.shuffle(deck)

        # Deal tableau (0..6 face-down, one face-up on each pile)
        self.tableau = []
        idx = 0
        for i in range(NUM_TABLEAU):
            down = deck[idx: idx + i]
            idx += i
            up = [deck[idx]]
            idx += 1
            self.tableau.append(Pile(face_down=down, face_up=up))

        # Remaining to stock (face-down)
        self.stock = deck[idx:]
        self.waste = []
        self.foundations = [0, 0, 0, 0]
        self.steps = 0

        obs = self._obs()
        info = {"action_mask": self._action_mask()}
        return obs, info

    # --------------- Step Logic ---------------
    def step(self, action: int):
        self.steps += 1
        reward = -0.01  # step penalty
        done = False
        truncated = False
        info: Dict = {}

        prev_foundation_sum = sum(self.foundations)
        prev_face_down = self._count_face_down()

        changed = self._apply_action(action)
        if not changed:
            reward -= 0.05  # illegal/no-op small penalty
        else:
            # Auto flip if needed
            flipped = self._auto_flip()
            if flipped:
                reward += 0.5

        # Foundation progress reward
        delta_found = sum(self.foundations) - prev_foundation_sum
        reward += 1.0 * delta_found

        # King-to-empty small reward (detected within _apply_action and returned flag?)
        # Simplify: heuristically reward if any empty pile exists AND a King is bottom of a moved run onto it.
        # We mark this in self._last_move_was_king_to_empty
        if getattr(self, "_last_move_was_king_to_empty", False):
            reward += 0.2
            self._last_move_was_king_to_empty = False

        if self._is_win():
            reward += 100.0
            done = True

        if self.steps >= self.step_limit:
            truncated = True

        obs = self._obs()
        info["action_mask"] = self._action_mask()
        info["face_down"] = self._count_face_down()
        info["foundation_sum"] = sum(self.foundations)
        return obs, reward, done, truncated, info

    # --------------- Action Decoding & Application ---------------
    def _apply_action(self, action: int) -> bool:
        """Return True if state changed; supports all masked actions."""
        # Draw from stock
        if action == ACTION_OFFSET_DRAW:
            return self._act_draw()

        # Waste -> Foundation
        if action == ACTION_OFFSET_WASTE_TO_FOUNDATION:
            return self._act_waste_to_foundation()

        # Waste -> Tableau[k]
        if ACTION_OFFSET_WASTE_TO_TABLEAU <= action < ACTION_OFFSET_TAB_TO_FOUNDATION:
            k = action - ACTION_OFFSET_WASTE_TO_TABLEAU
            return self._act_waste_to_tableau(k)

        # Tableau[i] -> Foundation
        if ACTION_OFFSET_TAB_TO_FOUNDATION <= action < ACTION_OFFSET_TAB_RUN_TO_TAB:
            i = action - ACTION_OFFSET_TAB_TO_FOUNDATION
            return self._act_tableau_to_foundation(i)

        # Tableau[i] run depth d -> Tableau[j]
        rem = action - ACTION_OFFSET_TAB_RUN_TO_TAB
        # Decode i, j, d from rem in row-major: for i in 0..6: for j in 0..6 if j!=i: for d in 1..13
        i, j, d = self._decode_tab_run_to_tab(rem)
        return self._act_tableau_run_to_tableau(i, j, d)

    def _decode_tab_run_to_tab(self, rem: int) -> Tuple[int, int, int]:
        per_src = (NUM_TABLEAU - 1) * 13
        i = rem // per_src
        rem2 = rem % per_src
        j_index = rem2 // 13
        d = (rem2 % 13) + 1
        # Map j_index (0..5) to actual j != i
        js = [x for x in range(NUM_TABLEAU) if x != i]
        j = js[j_index]
        return i, j, d

    # --------------- Specific Moves ---------------
    def _act_draw(self) -> bool:
        # Draw from stock to waste; if stock empty, recycle waste -> stock (face-down) preserving order
        if self.stock:
            self.waste.append(self.stock.pop())
            return True
        else:
            if not self.waste:
                return False  # nothing to recycle
            # Recycle: flip waste into stock (face-down), preserving order as in standard Klondike
            # Standard rule: turn the waste over to form a new stock (top of waste becomes bottom of new stock)
            self.stock = self.waste[::-1]
            self.waste = []
            return True

    def _act_waste_to_foundation(self) -> bool:
        if not self.waste:
            return False
        cid = self.waste[-1]
        suit, rank = id_to_card(cid)
        need = self.foundations[suit] + 1
        if rank == need:
            self.waste.pop()
            self.foundations[suit] += 1
            return True
        return False

    def _act_waste_to_tableau(self, k: int) -> bool:
        if not (0 <= k < NUM_TABLEAU) or not self.waste:
            return False
        cid = self.waste[-1]
        suit, rank = id_to_card(cid)
        pile = self.tableau[k]
        if self._can_place_on_tableau(cid, pile):
            self.waste.pop()
            self._place_on_tableau(cid, pile)
            if rank == 13 and len(pile.face_down) + len(pile.face_up) == 1:
                self._last_move_was_king_to_empty = True
            return True
        return False

    def _act_tableau_to_foundation(self, i: int) -> bool:
        if not (0 <= i < NUM_TABLEAU):
            return False
        pile = self.tableau[i]
        if not pile.face_up:
            return False
        cid = pile.face_up[-1]
        suit, rank = id_to_card(cid)
        need = self.foundations[suit] + 1
        if rank == need:
            pile.face_up.pop()
            self.foundations[suit] += 1
            return True
        return False

    def _act_tableau_run_to_tableau(self, i: int, j: int, depth: int) -> bool:
        if not (0 <= i < NUM_TABLEAU) or not (0 <= j < NUM_TABLEAU) or i == j:
            return False
        src = self.tableau[i]
        dst = self.tableau[j]
        # Must have enough face-up cards
        if depth <= 0 or depth > len(src.face_up):
            return False
        run = src.face_up[-depth:]
        # Validate run is descending alternating colors
        if not self._is_valid_run(run):
            return False
        # Check placement onto destination
        head = run[0]  # topmost of the moved run (closest to destination)
        if not self._can_place_on_tableau(head, dst):
            return False
        # Execute move
        src.face_up = src.face_up[:-depth]
        dst.face_up.extend(run)
        # King-to-empty hint
        suit, rank = id_to_card(head)
        if rank == 13 and (len(dst.face_up) + len(dst.face_down) == len(run)):
            # Moving a King-run onto an empty pile
            self._last_move_was_king_to_empty = True
        return True

    # --------------- Helpers ---------------
    def _auto_flip(self) -> bool:
        """Flip any face-down card that becomes newly exposed on tableau top. Returns True if any flipped."""
        flipped = False
        for pile in self.tableau:
            if not pile.face_up and pile.face_down:
                # If no face-up cards but has face-down, flip top face-down
                pile.face_up.append(pile.face_down.pop())
                flipped = True
        return flipped

    def _count_face_down(self) -> int:
        return sum(len(p.face_down) for p in self.tableau) + len(self.stock)

    def _is_win(self) -> bool:
        return sum(self.foundations) == 52  # all cards placed

    def _is_valid_run(self, run: List[int]) -> bool:
        if not run:
            return False
        # Check pairwise: rank desc by 1, colors alternate
        for a, b in zip(run, run[1:]):
            sa, ra = id_to_card(a)
            sb, rb = id_to_card(b)
            if ra != rb + 1:
                return False
            if is_red(sa) == is_red(sb):
                return False
        return True

    def _can_place_on_tableau(self, cid: int, pile: Pile) -> bool:
        suit, rank = id_to_card(cid)
        if pile.face_up:
            top = pile.face_up[-1]
            ts, tr = id_to_card(top)
            # Alternate color and descending rank
            return (is_red(suit) != is_red(ts)) and (rank == tr - 1)
        else:
            # Empty pile: only King
            return rank == 13

    def _place_on_tableau(self, cid: int, pile: Pile) -> None:
        pile.face_up.append(cid)

    # --------------- Observation & Masking ---------------
    def _obs(self) -> Dict[str, np.ndarray]:
        cards = np.full((NUM_TABLEAU, MAX_TABLEAU_HEIGHT), -1, dtype=np.int16)
        faceup = np.zeros((NUM_TABLEAU, MAX_TABLEAU_HEIGHT), dtype=np.int8)
        for i, pile in enumerate(self.tableau):
            stack = pile.face_down + pile.face_up
            h = min(MAX_TABLEAU_HEIGHT, len(stack))
            if h > 0:
                cards[i, :h] = np.array(stack[:h], dtype=np.int16)
                # face-up mask: last len(face_up) positions in stack are face-up
                fu = len(pile.face_up)
                if fu > 0:
                    faceup[i, h - fu: h] = 1
        waste_top = self.waste[-1] if self.waste else -1
        stock_count = len(self.stock)
        obs = {
            "tableau_cards": cards,
            "tableau_faceup": faceup,
            "foundations": np.array(self.foundations, dtype=np.int8),
            "waste_top": np.array([waste_top], dtype=np.int16),
            "stock_count": np.array([stock_count], dtype=np.int16),
            "steps": np.array([self.steps], dtype=np.int32),
        }
        return obs

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros((ACTION_SPACE_N,), dtype=bool)
        # Draw is always allowed (either draw or recycle)
        if self.stock or self.waste:
            mask[ACTION_OFFSET_DRAW] = True
        else:
            mask[ACTION_OFFSET_DRAW] = False

        # Waste -> Foundation
        if self._legal_waste_to_foundation():
            mask[ACTION_OFFSET_WASTE_TO_FOUNDATION] = True

        # Waste -> Tableau[k]
        if self.waste:
            for k in range(NUM_TABLEAU):
                if self._can_place_on_tableau(self.waste[-1], self.tableau[k]):
                    mask[ACTION_OFFSET_WASTE_TO_TABLEAU + k] = True

        # Tableau[i] -> Foundation
        for i in range(NUM_TABLEAU):
            if self._legal_tableau_to_foundation(i):
                mask[ACTION_OFFSET_TAB_TO_FOUNDATION + i] = True

        # Tableau runs i->j depth d
        idx = ACTION_OFFSET_TAB_RUN_TO_TAB
        for i in range(NUM_TABLEAU):
            src = self.tableau[i]
            if not src.face_up:
                idx += (NUM_TABLEAU - 1) * 13
                continue
            # Precompute all valid runs (start positions within face_up)
            # We consider depths 1..len(face_up) that form valid alternating runs ending at bottom
            valid_depths = self._valid_run_depths(src.face_up)
            js = [x for x in range(NUM_TABLEAU) if x != i]
            for j in js:
                dst = self.tableau[j]
                for d in range(1, 14):
                    allowed = d in valid_depths and d <= len(src.face_up)
                    if allowed:
                        head = src.face_up[-d]
                        if self._can_place_on_tableau(head, dst):
                            mask[idx] = True
                    idx += 1
        return mask

    def _valid_run_depths(self, face_up: List[int]) -> set:
        depths = set()
        # Walk upward from bottom and track longest alternating descending chain
        longest = 1
        depths.add(1)
        for k in range(len(face_up) - 1, 0, -1):
            a = face_up[k - 1]
            b = face_up[k]
            sa, ra = id_to_card(a)
            sb, rb = id_to_card(b)
            if (ra == rb + 1) and (is_red(sa) != is_red(sb)):
                longest += 1
            else:
                longest = 1
            depths.add(longest)
        # Also include all depths up to the longest contiguous valid chain ending at bottom
        # (agent can move any suffix of the valid chain)
        max_depth = max(depths) if depths else 1
        return set(range(1, max_depth + 1))

    def _legal_waste_to_foundation(self) -> bool:
        if not self.waste:
            return False
        cid = self.waste[-1]
        suit, rank = id_to_card(cid)
        return rank == self.foundations[suit] + 1

    def _legal_tableau_to_foundation(self, i: int) -> bool:
        pile = self.tableau[i]
        if not pile.face_up:
            return False
        cid = pile.face_up[-1]
        suit, rank = id_to_card(cid)
        return rank == self.foundations[suit] + 1

    # Convenience for quick random interaction
    def sample_legal_action(self, info: Optional[Dict] = None) -> int:
        mask = info.get("action_mask") if info is not None else self._action_mask()
        legal = np.nonzero(mask)[0]
        if len(legal) == 0:
            return 0  # fallback draw
        return int(random.choice(legal))

    # --------------- Render ---------------
    def render(self):
        lines = []
        lines.append(f"Steps: {self.steps}")
        lines.append(f"Foundations: " + ", ".join(f"{SUIT_NAMES[s]}:{r}" for s, r in enumerate(self.foundations)))
        lines.append(
            f"Stock: {len(self.stock)} | Waste top: {self._fmt_card(self.waste[-1]) if self.waste else 'None'}")
        lines.append("Tableau:")
        for i, p in enumerate(self.tableau):
            down = "[" + ",".join("XX" for _ in p.face_down) + "]"
            up = "[" + ",".join(self._fmt_card(c) for c in p.face_up) + "]"
            lines.append(f"  {i}: {down} {up}")
        return "\n".join(lines)

    def _fmt_card(self, cid: int) -> str:
        if cid < 0:
            return "--"
        s, r = id_to_card(cid)
        names = {1: "A", 11: "J", 12: "Q", 13: "K"}
        rr = names.get(r, str(r))
        return f"{rr}{SUIT_NAMES[s]}"

if __name__ == "__main__":
    env = KlondikeSolitaireEnv(seed=42)
    obs, info = env.reset()
    mask = info["action_mask"]
    obs, reward, terminated, truncated, info = env.step(env.sample_legal_action(info))
    print(env.render())