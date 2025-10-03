# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.5c3153xm9mha).

## Part 1
* Model accuracy: Validation Accuracy: 0.9739, Test Accuracy: 0.9677
* Link to saved model: https://drive.google.com/drive/u/6/folders/1c-85jfrs3_gYKizjWqU-3c8QvcR9r1au

* Free response:

4.A. 
What works well:
1. High overall accuracy: the model achieved 96.77% test accuracy and 97.39% validation accuracy. This shows that the Bi-LSTM architecture is very effective for POS tagging.

2. Common words and patterns: the model correctly tags most common words like determiners ('the', 'a'), pronouns ('I', 'me'), prepositions ('from', 'to'), and verbs ('should', 'would').

3. Proper noun recognition: the model excels at identifying proper nouns (NNP) like city names ('newark', 'los angeles', 'nashville', 'houston'), and it is crucial for flight-related queries.

4. Verb Forms: the model can correctly distinguishes between different verb forms (VB, VBZ, VBP) in context.

5. Plural Nouns: the model can accurately identifies plural nouns (NNS) like 'flights'.
         
What doesn't work as well:
1. Ambiguous words: words that can serve multiple grammatical functions are sometimes misclassified:
'list' (Sentence 7): used as a verb but tagged as proper noun (NNP):
'what' (Sentence 4): Wh-determiner (WDT) misclassified as wh-pronoun (WP)

2. Context-Dependent Tags: the model sometimes struggles with words whose POS depends heavily on context:
'a' (Sentence 5): in "t w a flight", 'a' should be part of the airline name (NNP) but was tagged as determiner (DT)



4.B. Analysis of incorrect tags:

1. Common Nouns → Proper Nouns (NN → NNP)
Example: 'airport' (Sentence 2)

why happens: when common nouns appear in sequences with proper nouns, the model gets contagion where it overgeneralizes the proper noun tag

Tendency: Common nouns in proper noun contexts get elevated to NNP


2. Wh-Determiners → Wh-Pronouns (WDT → WP)
Example: 'what' (Sentence 4)

Why it happens: the model can struggles to distinguish between interrogative word types based on syntactic context

Tendency: WDT (wh-determiner) gets misclassified as WP (wh-pronoun)



3. Proper Nouns → Determiners (NNP → DT)
Example: 'a' in "T W A" (Sentence 5)

Why it happens: Most likely is that extremely high-frequency words with strong default meanings resist reclassification in special contexts

Tendency: Rare proper noun uses of common words get defaulted to their most frequent tag



4. Verbs → Proper Nouns (VB → NNP)
Example: 'list' (Sentence 7)

Why happens: sentence-initial capitalization combined with following proper nouns creates strong NNP context

Tendency: Sentence-initial verbs get misclassified as proper nouns due to capitalization



4.C. Micro- vs Macro-Level Tag Errors:

Micro-Level Errors (less severe):

NN vs NNP ('airport'): Both are nouns, so syntactic parsing might still work

WDT vs WP ('what'): Both are interrogative words

Macro-Level Errors (more severe):

VB vs NNP ('list'): Verb vs proper noun - completely different syntactic roles

NNP vs DT ('a'): Proper noun vs determiner - major syntactic function difference

Which is worse?
I think that macro-level errors are significantly worse because they represent fundamental misunderstandings of syntactic structure. A verb being tagged as a proper noun (like 'list' → NNP) could completely break downstream parsing, while NN/NNP confusion might only affect named entity recognition.





## Part 2
* How many unique rules are there? 
Total unique rules: 419
* What are the top five most frequent rules, and how many times did each occur?
Top 5 most frequent rules:
   1. IN -> IN_t : 482 occurrences
   2. PUNC -> PUNC_t : 469 occurrences
   3. NP_NNP -> NP_NNP_t : 451 occurrences
   4. NNP -> NNP_t : 408 occurrences
   5. NN -> NN_t : 281 occurrences
* What are the top five highest-probability rules with left-hand side NNP, and what are their probabilities?
c. Top 5 highest probability NP rules:
   1. NP -> NNP NNP : 0.1928
   2. NP -> NP NP* : 0.1159
   3. NP -> DT NN : 0.1014
   4. NP -> DT NNS : 0.0855
   5. NP -> DT NP* : 0.0841

* Free Response: Did the most frequent rules surprise you? Why or why not?

The most frequent rules did not surprise me. All top five rules are terminal rules (mapping POS tags to terminal symbols), which makes sense because every word in the treebank is associated with a POS tag. Tags like IN (prepositions), PUNC (punctuation), NNP (proper nouns), and NN (common nouns) are very common in English text, so their high frequency is logical. 

## Part 3
* CKY parses using gold POS tags:
* CKY parses using predicted POS tags:
* Free response:
