"""
Goal: Highest-probability parse using a PCFG, with POS tags (from your BiLSTM)
      as the terminal layer.

Inputs:
  - cfg / counts / probs: representations of your PCFG
  - true POS tags or POS tags as the output of a trained BiLSTM POS tagger

Output:
  - For each sentence line from stdin or a file, print one bracketed tree to stdout
    (or an empty line if no parse) with its probability.

Keep these three phases conceptually separate:
  (1) POS tagging + diagonal initialization
  (2) CKY dynamic program over spans (big nested loop)
  (3) Root selection + backpointer reconstruction + printing

You are free to choose your exact data structures, as long as you can:
  - store best scores for labels over spans
  - remember how each best item was built (backpointers)
  - reconstruct a bracketed tree string at the end

-----------------------------------------------------------------------------
0) GRAMMAR + PROBABILITIES + <unk>
-----------------------------------------------------------------------------
* When reading a sequence of POS tags, map each tag to itself if in vocab, 
  or else to a special token like "<unk>" (but keep original for printing).
* After reconstruction, print the original word as the leaf instead.
* Use log probabilities to avoid underflow:
  score = log P(A -> B C) + score(left) + score(right)

-----------------------------------------------------------------------------
1) READ ONE SENTENCE LINE → TOKENS → POS TAGS → DIAGONAL INIT
-----------------------------------------------------------------------------
- Read lines from train.pos as (word, tag) tuples using read_pos_files().
- Run your BiLSTM POS tagger on the words to get one POS per token.
  * Hint: to confirm that CKY works, first just use the true POS tags rather than running it through your tagger.

- Create two core tables for CKY (choose your own structures, examples below):
    chart:   stores best scores for labels over spans
             indexable by span (i, k) and then by label
    backptr: stores how that best label@span was formed
             (for terminals: the terminal tag; for binary: (left_label, split_index, right_label))

  Example shapes (you can pick others):
    chart[(i, k)][label]  -> best_score (log-prob or prob)
    backptr[(i, k)][label] -> for terminal: stored tag
                              for binary: (left_label, j, right_label)

- Diagonal initialization (length-1 spans [i, i+1)):
    For each position i:
      * Use the POS tag(s) for token i as candidate preterminals.
      * Record best scores per POS at (i, i+1).
      * Record backptr so reconstruction can print "(POS word)".

-----------------------------------------------------------------------------
2) CKY DYNAMIC PROGRAM (THE BIG NESTED LOOP)
-----------------------------------------------------------------------------
The standard CKY fill uses three nested loops over span length, start index,
and split point. Conceptually:

  for span_length in 2..n:
    for i in 0..(n - span_length):
      k = i + span_length
      initialize chart[(i, k)] and backptr[(i, k)] (empty)

      for j in (i+1)..(k-1):   # split index
        # Consider all ways to combine a left piece (i, j) with a right piece (j, k)
        for each left_label in chart[(i, j)]:
          for each right_label in chart[(j, k)]:
            # Check if any rule A -> left_label right_label exists in your PCFG
            for each A with P(A -> left_label right_label):
              candidate_score = chart[(i, j)][left_label] + \
                                chart[(j, k)][right_label] + \
                                log P(A -> left_label right_label)
              if candidate_score is better than current chart[(i, k)][A]:
                  update chart[(i, k)][A] = candidate_score
                  set backptr[(i, k)][A] = (left_label, j, right_label)

Notes:
  - Only binary rules are considered here (CNF).
  - Keep everything in log-space abd use addition rather than multiplication.

-----------------------------------------------------------------------------
3) ROOT SELECTION, RECONSTRUCTION, PRINTING
-----------------------------------------------------------------------------
- After the table is filled, focus on the full span (0, n).
  * Prefer the designated start symbol (e.g., 'TOP') if present at (0, n).
  * If 'TOP' is not present, produce an empty parse.

- Reconstruct the tree via backpointers:
  * Define a recursive function:
      reconstruct(label, i, k):
        bp = backptr[(i, k)][label]
        if bp is a terminal word (or "<unk>"):
            return "(label word_or_original)"
        else:
            (left_label, j, right_label) = bp
            left_subtree  = reconstruct(left_label,  i, j)
            right_subtree = reconstruct(right_label, j, k)
            return f"(label {left_subtree} {right_subtree})"

  * Ensure terminals print the original word here rather than POS tags.

- Output:
  * Print the bracketed tree string for each input sentence (or an empty line
    if no parse), one sentence per line.
  * Print the final score for TOP at (0, n) as a log-prob.

-----------------------------------------------------------------------------
4) PRACTICAL TIPS / DECISIONS (YOU CHOOSE)
-----------------------------------------------------------------------------
- Data structures:
    * dict-of-dicts is fine; you can also use defaultdicts.
    * You can index chart/backptr by tuples (i, k) or use a 2D list.

- Efficiency:
    * Iterate only over labels that actually occur in the subspans.
    * If your PCFG is stored by RHS (B, C) -> {A: prob}, you can quickly find
      candidate parents A for a given pair (B, C).

- Scores:
    * Prefer log-space: add logs instead of multiplying probabilities.

- Debugging:
    * Print the diagonal cells after initialization to verify POS entries.
    * For a tiny sentence (2–3 words), print intermediate chart cells per length.
    * If reconstruction fails, check that backptr entries are actually written
      whenever you write a score.

-----------------------------------------------------------------------------
5) MINIMUM I/O LOOP
-----------------------------------------------------------------------------
for each line from stdin:
  tokens = line.split()
  tags = run_pos_tagger(orig_tokens)

  initialize empty chart/backptr

  # diagonal init
  for i in range(n):
    fill chart[(i, i+1)] and backptr[(i, i+1)] with POS entries

  # CKY nested loops (length, start i, split j) using binary PCFG rules
  fill chart/backptr for spans of length >= 2

  if TOP in chart[(0, n)]:
      tree_str = reconstruct('TOP', 0, n)   # bracketed
      print(tree_str)
      print(logprob_of_TOP_to_stderr)
  else:
      print("")  # empty line if no parse
"""
