# Methodology

The technical details behind the articles: how one game is played, how hesitation and pushing are
modeled, how the random numbers are drawn, how the simulator is validated, and what each experiment
tests. Every parameter named here lives in `Params` (`src/glass_bridge/params.py`).

## 1. The game

- **Bridge:** `n_steps = 18` steps. Each step has two panes; one is safe, chosen 50/50 and independently.
- **Players:** `n_players = 16`, in a fixed queue. Player 1 goes first.
- **Follow the leader:** players walk in single file behind the *leader* (the player at the front).
  Nobody has to remember the path, so there are no memory errors: a broken pane shows everyone
  where the safe one is.
- **Frontier:** the furthest step whose safe pane is known. Steps up to the frontier are walked
  without hesitation.
- **Clock:** `clock = 960` s (16 minutes). Anyone who hasn't left the bridge by then is eliminated.

## 2. One turn

1. **Walk up.** The leader walks to the frontier, `t_move = 3` s per known step.
2. **Hesitate.** The leader draws a decision time (section 3).
3. **Push check** (only if the player behind is a pusher; section 4).
4. **Jump.** The leader's guess is a coin flip. Right: they stand on the new step. Wrong: they fall.
   Either way the pane is revealed and the frontier moves up one step.
5. **After a fall,** the next player, who was standing right behind, becomes the leader.
6. **End of the bridge.** When the frontier reaches step 18, the leader walks off and each remaining
   player follows `t_move` behind the one in front. Anyone exiting after the clock is a timeout.

Jumps themselves take no time; time passes while walking, hesitating and pushing. Only the gap is
walked: after a fall, the next player was standing one step behind the fallen leader, so they walk
the 2 steps to the newly revealed pane, not the whole known path.

**Outcomes** (one per player): survived, glass (chose the breaking pane), pushed, drag, backfire,
timeout. The event log (`trace(params, seed, game)`) records every walk, hesitation, push, jump,
fall, exit and timeout with its time.

## 3. Hesitation

Only a leader facing an **unknown** pane hesitates. The decision time is random:

- **Distribution:** lognormal by default, with **median** `d0` and spread `sigma` (the standard
  deviation of the log). With `sigma = 1`, one decision in twenty takes about 5.2 times the median.
- **"Typical decision time"** in the articles means the **median** `d0`: half of decisions are faster.
  It is not the average: with `sigma = 1` the mean is exp(1/2) ≈ 1.65 times the median, so a typical
  40 s decision corresponds to an average of about 66 s, pulled up by the occasional long freeze.
- **Pressure** (`beta`): the median becomes `d0 · exp(beta · u)`, where `u` is the share of the clock
  already used when the leader reaches the pane. `beta < 0`: players rush as time runs out; `beta = 0`:
  constant hesitation; `beta > 0`: they freeze. Rush, no pressure and freeze are `beta = −1.5, 0, +1.5`.
- **Robustness:** `delay = "gamma"` uses a gamma distribution with the same median and the same mean
  as the lognormal, driven by the same random draws.

Article 1 uses constant hesitation (`beta = 0`, experiment E9), so its "typical decision time" is the
same all game.

## 4. Pushing (Article 2)

- **Who:** exactly `n_pushers` players, placed at random in the queue. Only the player directly
  behind the leader can push.
- **When:** never before `t_start = 420` s (the 7-minute rule). After that, a pusher pushes a
  leader who has hesitated longer than `patience = 15` s, if the push can finish before the leader
  would have jumped and before the clock runs out.
- **The struggle** takes `t_push = 3` s and ends one of three ways: the push **lands** (`p_land = 0.6`,
  the victim goes over), a **drag** (`p_drag = 0.2`, both go over) or a **backfire**
  (`p_backfire = 0.2`, the pusher falls and the leader keeps hesitating).
- **Pane contact:** whoever goes over touches the next pane with probability `p_contact`
  (default 0.5). With contact, the pane is revealed; a pushed victim who lands on the safe pane survives
  one step ahead. Without contact, the victim falls clear and nothing is revealed.
- **Paired test** (E10): `focal` sets one queue position aside; `focal_pushes` makes that player a
  pusher (1) or not (0) while the other pushers and every random draw stay the same.

## 5. Random numbers: same bridges in every scenario

Every random number a game needs is drawn up front, each kind from its own stream (`draws.py`):
the bridge, the leaders' guesses, the hesitation draws, the pusher order, the push struggles and
the pane contacts. The streams come from one seed through NumPy's `SeedSequence`.

So game *g* under a given seed has the same bridge, guesses and hesitation draws in every
scenario, and adding a pusher keeps the earlier pushers in place (common random numbers).
Differences between settings come from the rules, not from luck. Each game's draws don't depend on
how many games are run.

## 6. Validation

- **Ideal play** (`BASELINE`: no clock, no pushers) must reproduce the exact odds. Player *k*
  survives if at most *k* − 1 of the 18 guesses are wrong (binomial), which gives exactly
  7.0000763 expected survivors; Tijms' Markov chain gives the same distribution. The simulation
  matches by position and in distribution within 4 standard errors (tests), and the 100,000-game
  baseline (E0) gives 6.995 ± 0.007.
- **Rule checks** (`tests/`): forced pushes that always reveal a pane don't change survival without a
  clock; a tiny clock kills everyone; tiny bridges behave; nobody pushes before `t_start`; every player
  ends exactly once; invalid settings are rejected.
- **Timing** (`tests/test_timing.py`, hand-built draws so every time is exact): a perfect crossing exits
  at 183 s, then every 3 s; with a 200 s clock the first six exit and ten time out; reaching step 18
  at 180 s with a 182 s clock is not enough (stepping off takes 3 s); a decision that would end after
  the clock means a timeout; after a fall the next leader walks only the 2-step gap.
- **Reproducibility:** shared draws across scenarios, `trace` replays any batch game, and re-running a
  sweep gives a byte-identical results file.

## 7. Experiments

All experiments use seed 2026 and 100,000 games per setting: 1,181 settings, 118.1 million games.
Article 1 rests on E0, E9 and the robustness runs E3, E5 and E7: 503 settings, 50.3 million games
(15.6 million of them in E9, the main result). The rest are pushing experiments for Article 2.

| ID | Config | Question | Used in |
|---|---|---|---|
| E0 | `E0_baseline` | Does ideal play match the exact odds? | Article 1, fig 1 |
| E1 | `E1_pushers` | Pushers (0–8) × pressure × pane contact (0–1) | Article 2, figs 2–4 |
| E3 | `E3_clock` | Time limit (10–25 min) × pressure × {0, 5} pushers | fig 5 (repo) |
| E4 | `E4_push_start` | When pushing starts (0–12 min) | repo |
| E5 | `E5_sensitivity` | One parameter at a time: `d0`, `sigma`, `t_move`, `patience`, push split, gamma | robustness |
| E7 | `E7_hesitation_dial` | Pressure from rush to freeze in 25 steps | fig 7 (repo) |
| E8 | `E8_full_contact_sensitivity` | E5 with every push revealing a pane | Article 2 robustness |
| E9 | `E9_clock_x_decision_time` | Constant hesitation 5–90 s × time limit 8–30 min | Article 1, figs 6, 8, 10 |
| E10 | `E10_paired_pusher` | Same player pushing vs not, at each position | Article 2, fig 9 |

E2 and E6 are analyses of E1 and E3, not separate runs.

```bash
.venv/Scripts/python -m glass_bridge.experiments configs/E9_clock_x_decision_time.toml
.venv/Scripts/python -m glass_bridge.figures      # needs: pip install -e ".[figures]"
```

## 8. Outputs

`results/<experiment>.csv` has one row per setting: every parameter, `games`, `seed`, then

- `mean_survivors`, `se` (Monte Carlo standard error of the mean)
- `p_0_survivors` … `p_16_survivors`: the survivor distribution
- `survival_pos_1` … `survival_pos_16`: survival by queue position
- `share_<outcome>`: share of all players ending each way (× 16 = players per game)
- `pusher_survival`, `nonpusher_survival`
- `p_clock_ran_out`: share of games in which at least one player timed out

`results/<experiment>.json` records the seed, git commit, whether the code had uncommitted changes
(`dirty`), the run time and the Python, NumPy and SciPy versions. `figures/SOURCES.md` lists each
figure's data source.

## 9. Limitations

- Decision times, pressure and push mechanics are assumptions, varied in sensitivity runs, not
  measurements of real people.
- The model has no overtaking, no memory errors, no strategies chosen by players, and no tricks such as
  inspecting the glass.
- Results are specific to these rules (18 steps, 16 players, single file). Monte Carlo precision
  (standard errors are tiny at 100,000 games) does not remove model uncertainty.
