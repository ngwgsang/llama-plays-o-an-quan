Board: a ring of 12 cells. Clockwise order is
  QA -> A1 -> A2 -> A3 -> A4 -> A5 -> QB -> B5 -> B4 -> B3 -> B2 -> B1 -> (back to QA)
QA and QB are Mandarin cells (Quan). Team A owns A1-A5, team B owns B1-B5.

Basic rules:
- Scattering: pick one of YOUR non-empty cells (never a Mandarin cell) and a direction
  (clockwise or counter_clockwise). All peasants in that cell are dropped one by one
  into the following cells, wrapping around the board.
- If the cell right after the last dropped peasant is a non-empty regular cell,
  pick up its peasants and keep scattering in the same direction.
- If the cell right after the last dropped peasant is a Mandarin cell, the turn ends.
- Capturing: if the next cell is empty and the cell beyond it has pieces, you capture
  all pieces in that cell. Peasant = 1 point, Mandarin = 10 points.
- The game ends when both Mandarins are captured or no legal moves remain.
  The player with the higher score wins.

Extra rules:
- E1 Immature Mandarin: a Mandarin cell with fewer than 5 peasants cannot be captured.
- E2 Forced Redistribution: if all your cells are empty at the start of your turn, you
  must return 5 captured points (one peasant per cell) to continue. If you cannot, you lose.
- E3 Early Game Restriction: no Mandarin captures are allowed during the first two rounds.
- E4 Two-Empty Rule: if the two cells immediately after the last scattered peasant are
  both empty, your turn ends immediately.
- E5 Forced Capture Chain: after a capture, if the next cell is empty and the one beyond
  it has pieces, you must capture again, as long as valid captures exist.
