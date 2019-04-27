# Detection of negation in sentences

A NLP Model that detects **cues (words that cause negation)** and **scope (negated part of the sentence)**.

# Evaluation results

<img width="644" alt="eval1" src="https://user-images.githubusercontent.com/29833297/56854442-37738e00-6954-11e9-9bd9-e8c0539f66c7.PNG">

<img width="632" alt="eval2" src="https://user-images.githubusercontent.com/29833297/56854443-393d5180-6954-11e9-9b28-12b1b137b851.PNG">

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

<img width="246" alt="scopeisnt" src="https://user-images.githubusercontent.com/29833297/56854516-8f5ec480-6955-11e9-8f03-e5b5519b8688.PNG"

- Features selected for classification of the scopes are:
		- Token
		- Lemma
		- Part of Speech tag (PoS)
		- PoS of next and previous words
		- Graph distance of token to cue
		- Dependency path from token to cue
		- PoS of the cue
		- Cue type
		
- Distance from token to cue is calculated using Dijkstra's algorithm and treated as a separate feature. In the following figure, distance between the token “telegram” to cue “No” and is 3.
- Dependency path is the shortest path between token and cue, containing all the dependency relations of the words along the path.
The dependency path from token “telegram” to cue “No” is:
			*\dobj/nsubj/neg*
		
		- ‘\’ represents moving up the tree
		- ‘/’ represents moving down the tree

# Vectorization

- After generating the dictionaries and extracting the features, we convert our feature dictionaries into binary vectors.  
- Each distinct feature value is treated as a separate attribute. A binary vector is created for each instance based on the attributes that it contains.
- To do the vectorisation, we use DictVectorizer from the SCIKIT-LEARN toolkit. This module creates a vector for each feature and then concatenates them so we end up with a single feature vector for each instance.
- For cues, we vectorise the data space for each potential cue token.
- For scopes, we vectorise the data space for each sentence that contains a cue.

# Classifiers used


![WhatsApp Image 2018-12-16 at 10 09 58 PM](https://user-images.githubusercontent.com/29833297/56854447-44907d00-6954-11e9-9341-0a21e1b4acab.jpeg)

- **For cue classification**
	The classification of the cues was looked at as a binary classification, as we can identify cues in isolation, irrespective of the sentences that they were part of. 
We considered using SVM classifiers for this classification. SVC works by creating a hyperplane that divides the data space into the 2 classes.
However, we realised that there were instances where some cues did not always act as cues, which resulted in overlapping of the two classes. Hence we decided on using a kind of penalty parameter that would create a *less stricter classifier than SVC*.
Therefore, we implemented the **NSlackSSVM** which introduced slack variables which relaxes the constraints on the hyperplane by implementing marginal hyperplanes. The error for a training data point on the wrong side of the marginal hyperplane was calculated which helped determine the ideal hyperplane.

- **For scope classification**
The sentence can be interpreted as a sequence of tokens, hence the task of scope resolution is to label each token in the sentence as in or out of scope. For sequence labelling, statistical models have shown to be useful because the label of a state i in a sequence typically depends on the i-1 already observed states. 
The output classes are the 4 labels provided to these scopes.
It is a discriminative model which contains a compatibility function which uses approaches like **Conditional Random Field (ChainCRF)**
We used the **FrankWolfeSSVM** for the structural prediction of the scopes in the form of sentences.



