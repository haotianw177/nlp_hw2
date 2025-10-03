import trees
import fileinput
import collections
import re
"""You should not need any other imports, but you may import anything that helps."""

counts = collections.defaultdict(collections.Counter)
probs = {}  # Dictionary to store probabilities
cfg = {}    # Dictionary to store CFG rules

def process_tree(tree_str):
    """Process a single tree string and count all rules"""
    try:
        tree = trees.Tree.from_str(tree_str)
        if tree.root is not None:
            # Remove empty nodes and unit productions for cleaner rules
            tree.remove_empty()
            tree.remove_unit()

            # Traverse tree bottom-up to collect rules
            for node in tree.bottomup():
                if node.children:  # If node has children (not a leaf)
                    lhs = node.label  # Left-hand side of rule

                    # Handle terminal vs non-terminal rules
                    if len(node.children) == 1 and not node.children[0].children:
                        # Terminal rule: POS tag -> word (but we use POS_t format)
                        pos_tag = node.label
                        rhs = (pos_tag + '_t',)  # Convert to tuple for hashability
                    else:
                        # Non-terminal rule
                        rhs = tuple(child.label for child in node.children)

                    # Count this rule occurrence
                    counts[lhs][rhs] += 1
    except Exception as e:
        print(f"Error processing tree: {e}", file=sys.stderr)

def compute_probabilities():
    """Compute conditional probabilities for each rule"""
    global probs
    probs = {}

    for lhs in counts:
        probs[lhs] = {}
        total = sum(counts[lhs].values())  # Total count for this LHS

        for rhs in counts[lhs]:
            # P(RHS | LHS) = count(LHS->RHS) / total_count(LHS)
            probs[lhs][rhs] = counts[lhs][rhs] / total

def print_pcfg():
    """Print the PCFG in readable format"""
    print("PCFG Rules:")
    print("-" * 50)

    # Sort LHS symbols alphabetically for consistent output
    for lhs in sorted(probs.keys()):
        for rhs in sorted(probs[lhs].keys()):
            # Format RHS for display
            if len(rhs) == 1 and rhs[0].endswith('_t'):
                rhs_str = rhs[0]  # Terminal rule
            else:
                rhs_str = ' '.join(rhs)  # Non-terminal rule

            probability = probs[lhs][rhs]
            print(f"{lhs} -> {rhs_str} # {probability:.4f}")

def generate_report():
    """Generate the required report for part 3"""
    print("\n" + "="*60)
    print("REPORT")
    print("="*60)

    # 3a. Count unique rules
    total_rules = sum(len(counts[lhs]) for lhs in counts)
    print(f"a. Total unique rules: {total_rules}")

    # 3b. Top 5 most frequent rules
    print("\nb. Top 5 most frequent rules:")
    all_rules = []
    for lhs in counts:
        for rhs in counts[lhs]:
            all_rules.append((lhs, rhs, counts[lhs][rhs]))

    # Sort by frequency descending
    all_rules.sort(key=lambda x: x[2], reverse=True)

    for i, (lhs, rhs, count) in enumerate(all_rules[:5]):
        rhs_str = rhs[0] if len(rhs) == 1 and rhs[0].endswith('_t') else ' '.join(rhs)
        print(f"   {i+1}. {lhs} -> {rhs_str} : {count} occurrences")

    # 3c. Top 5 highest probability NP rules
    print("\nc. Top 5 highest probability NP rules:")
    if 'NP' in probs:
        np_rules = []
        for rhs in probs['NP']:
            np_rules.append((rhs, probs['NP'][rhs]))

        # Sort by probability descending
        np_rules.sort(key=lambda x: x[1], reverse=True)

        for i, (rhs, prob) in enumerate(np_rules[:5]):
            rhs_str = rhs[0] if len(rhs) == 1 and rhs[0].endswith('_t') else ' '.join(rhs)
            print(f"   {i+1}. NP -> {rhs_str} : {prob:.4f}")
    else:
        print("   No NP rules found")

    # 3d. Free response placeholder
    print("\nd. Free Response: [Your observations about the most frequent rules]")

def build_cfg_structure():
    """Build CFG structure for potential CKY implementation"""
    global cfg
    cfg = {}

    for lhs in probs:
        for rhs in probs[lhs]:
            # Store rules indexed by RHS for easier CKY lookup
            if rhs not in cfg:
                cfg[rhs] = []
            cfg[rhs].append((lhs, probs[lhs][rhs]))

def main():
    """Main function to process training data and generate PCFG"""
    # Read and process all trees from train.trees
    print("Reading and processing trees from train.trees...")

    tree_count = 0
    current_tree = ""

    for line in fileinput.input('train.trees'):
        line = line.strip()
        if line:
            current_tree += line
            # Check if we have a complete tree (balanced parentheses)
            if current_tree.count('(') == current_tree.count(')'):
                process_tree(current_tree)
                tree_count += 1
                current_tree = ""

    print(f"Processed {tree_count} trees")

    # Compute probabilities
    compute_probabilities()

    # Build CFG structure
    build_cfg_structure()

    # Print the PCFG
    print_pcfg()

    # Generate the report
    generate_report()

if __name__ == "__main__":
    import sys
    # Remove the NotImplementedError and run main
    main()