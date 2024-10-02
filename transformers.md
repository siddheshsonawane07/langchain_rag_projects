<h1> Transformers Theory </h1>

Transformers refer to a type of deep learning model introduced in 2017.
It is widely used in NLP tasks such as transation, summarization, and text generation.

Key Features of Transformers
1. Self Attention Mechanism
2. Encoder Decoder Architecture
3. Parallel Processing
4. Positional Encoding

<h1>Self Attention Mechanism in Transformers</h1>
The self-attention mechanism is a core concept in Transformers, allowing a model to focus on different parts of an input sequence when processing each word or token. This mechanism helps the model capture dependencies between words, even if they are far apart in the sequence.

Example Sentence:
"The cat sat on the mat."

The goal of self-attention is to figure out which words in this sentence are important when processing each individual word.

Step 1: Input Representation Each word in the sentence is represented as an embedding vector. For simplicity, we'll assume each word has been converted into a numerical vector that captures its meaning.
["The", "cat", "sat", "on", "the", "mat"]


Step 2: Query, Key, and Value Vectors For each word, three vectors are generated:
Query (Q): What this word is looking for in other words.
Key (K): How relevant this word is to the others.
Value (V): The actual information contained in the word.
These vectors are derived by multiplying the word embedding by learned weight matrices.

Step 3: Calculating Attention Scores To figure out how much attention one word should pay to other words, we calculate the dot product between the Query vector of a word and the Key vectors of all other words. This gives us attention scores that indicate how much focus should be placed on each word.
Let's take the word "sat" as an example. We'll compute the attention scores for "sat" relative to all the words in the sentence.
Attention scores for "sat":
Q(sat) · K(The) → Score 1
Q(sat) · K(cat) → Score 2
Q(sat) · K(sat) → Score 3 (this score is typically highest)
Q(sat) · K(on)  → Score 4
Q(sat) · K(the) → Score 5
Q(sat) · K(mat) → Score 6
These scores determine how much "sat" should attend to each word, including itself.


Step 4: Applying Softmax The attention scores are passed through a softmax function to convert them into probabilities, which represent how much focus (or attention) is given to each word. Higher scores mean more attention.
Softmax(Score 1, Score 2, ..., Score 6) → [0.1, 0.3, 0.5, 0.05, 0.03, 0.02]
In this case, "sat" pays most attention to itself (the highest value, 0.5) and a fair amount of attention to "cat" (0.3), but much less to other words.


Step 5: Weighted Sum of Values Finally, the attention probabilities are used to compute a weighted sum of the Value (V) vectors of all the words. This sum gives us a new representation for the word "sat", which now incorporates information from the other words it focused on.
New representation of "sat" = 0.1 * V(The) + 0.3 * V(cat) + 0.5 * V(sat) + 0.05 * V(on) + 0.03 * V(the) + 0.02 * V(mat)

This process is repeated for every word in the sentence. Each word gets a new representation that captures its relationships with other words, allowing the model to understand context and long-range dependencies.

<h1>Encoder Decoder Architecture</h1>
Encoder: Takes the input sequence (e.g., a sentence) and compresses it into a fixed-size internal representation. It consists of multiple layers of self-attention and feed-forward networks (in the case of a Transformer).

Decoder: Uses this internal representation to generate an output sequence (e.g., a translated sentence or a summary).

Step-by-Step Process

English Sentence → [Encoder] → Encoded Representation → [Decoder] → French Sentence

Step 1: The encoder takes in the entire input sequence at once. For example, in machine translation, the input might be a sentence in English:
    "The cat sat on the mat."

Self-Attention: Each word in the input attends to other words in the same sentence to capture relationships. For instance, "sat" might pay attention to "cat" because they're related.

Positional Encoding: Since the Transformer encoder does not inherently understand the order of words (unlike RNNs), it adds positional encodings to the input embeddings to represent the order of the words.

The encoder then produces a contextual representation of the entire sentence (a set of vectors, one for each word). These vectors encode both the meaning of each word and its relationship to other words in the sentence.


Step 2: The decoder generates the output sequence, such as a translation of the English sentence into another language, say French.
It operates step by step, predicting one word at a time. For example:
    "Le chat s'est assis sur le tapis."
Self-Attention in the Decoder: When generating a word, the decoder looks at the previous words it has already generated. For instance, when predicting the word "assis," it attends to "chat" because "chat" is relevant to "assis" (the cat is sitting).

Encoder-Decoder Attention: The decoder also attends to the encoded input sentence from the encoder. When predicting each word, it consults the encoded input to find relevant parts of the original sentence. For instance, when generating "chat," it attends to the English word "cat" in the input.


Step 3: At each step, the decoder outputs one word in the target language. Once a word is generated, it feeds back into the decoder for the next step (this is called autoregressive decoding).

The process continues until the decoder generates the special end-of-sequence token (e.g., <EOS>), signaling that the output sequence is complete.




Each of these parts can be used independently, depending on the task:

Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
Decoder-only models: Good for generative tasks such as text generation.
Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.