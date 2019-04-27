# Detection of negation in sentences

A NLP Model that detects **cues (words that cause negation)** and **scope (negated part of the sentence)**.

# Dataset

Data included text from Arthur Conan Doyle's famous book *Hounds of Baskerville*. The text was stored in the CONLL format with the following column descriptions.
- Column 1: book name
- Column 2: sentence number
- Column 3: token number
− Column 4: word 
− Column 5: lemma of the word 
− Column 6: Part of Speech tag of the word
- Column 7: parse tree information 
Column 8: The negation information 
Columns after index 8 follow the pattern of :
[Cue, Scope, Negated event] correspondingly.

<img width="763" alt="beforeprocess" src="https://user-images.githubusercontent.com/29833297/56854437-27f44500-6954-11e9-8268-fa51296e3adf.PNG">

# Preprocessing

We took the following steps to pre-process the data into a favourable format:

- Converted the dataset into a raw format by extracting the tokens from each line (at index 3).
- We then implemented the **StandfordCoreNLP Dependency parser**, that parsed the data and created a dependency tree for each sentence, which created dependency relations between words, that helped us determining scopes.

# Feature extraction for cues

The sentences are stored in the form of dictionaries. Each word in a sentence, is treated as a separate entity with all its features as key-value pairs. Corresponding cues and scopes are stored for each sentence in the dictionary.

<img width="300" alt="cue_dict" src="https://user-images.githubusercontent.com/29833297/56854439-32164380-6954-11e9-9d09-83bea4e177c0.PNG">

- First step is to obtain a dictionary of all possible cues in the training dataset, as well as their respective affix [prefix, suffix, infix] cues.
- Cues that do are not affixes are denoted by ‘s’ and those with affixes are denoted by ‘a’.
- Features selected for classification of the cues are:
		- Token 
		- Lemma 
		- Part of Speech tag
		- Previous and Next words
		- Character 5-bigrams, for afﬁxal cues
  
Cue labels are assigned to each cue instance, based on their occurrence in the dataset. Treated as a **binary classification problem where label 1 (cue) and -1 (non-cue)**.

# Feature extraction for scope

- Features selected for classification of the scopes are:
		- Token
		- Lemma
		- Part of Speech tag (PoS)
		- PoS of next and previous words
		- Graph distance of token to cue
		- Dependency path from token to cue
		-  PoS of the cue
		- Cue type
		
- Distance from token to cue is calculated using Dijkstra's algorithm and treated as a separate feature. In the following figure, distance between the token “telegram” to cue “No” and is 3.
- Dependency path is the shortest path between token and cue, containing all the dependency relations of the words along the path.
The dependency path from token “telegram” to cue “No” is:
			*\dobj/nsubj/neg*
		- ‘\’ represents moving up the tree
		- ‘/’ represents moving down the tree
	





