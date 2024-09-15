# LLM Production

## Introduction to LLMs

## What are Large Language Models

By now, you might have heard of them. Large Language Models, commonly known as LLMs, are a sophisticated type of neural network. These models ignited many innovations in the field of natural language processing (NLP) and are characterized by their large number of parameters, often in billions, that make them proficient at processing and generating text. They are trained on extensive textual data, enabling them to grasp various language patterns and structures. The primary goal of LLMs is to interpret and create human-like text that captures the nuances of natural language, including syntax (the arrangement of words) and semantics (the meaning of words).

The core training objective of LLMs focuses on predicting the next word in a sentence. This straightforward objective leads to the development of emergent abilities. For example, they can conduct arithmetic calculations, unscramble words, and have even demonstrated proficiency in professional exams, such as passing the [US Medical Licensing Exam](https://healthitanalytics.com/news/chatgpt-passes-us-medical-licensing-exam-without-clinician-input). Additionally, these models have significantly contributed to various NLP tasks, including machine translation, natural language generation, part-of-speech tagging, parsing, information retrieval, and others, even without direct training or fine-tuning in these specific areas.

The text generation process in Large Language Models is autoregressive, meaning they generate the next tokens based on the sequence of tokens already generated. The attention mechanism is a vital component in this process; it establishes word connections and ensures the text is coherent and contextually appropriate. It is essential to establish the fundamental terminology and concepts associated with Large Language Models before exploring the architecture and its building blocks (like attention mechanisms) in greater depth. Let’s start with an overview of the architecture that powers these models, followed by defining a few terms, such as language modeling and tokenization.

# Key LLM Terminologies

## The Transformer
The foundation of a language model that makes it powerful lies in its architecture. Recurrent Neural Networks (RNNs) were traditionally used for text processing due to their ability to process sequential data. They maintain an internal state that retains information from previous words, facilitating sequential understanding. However, RNNs encounter challenges with long sequences where they forget older information in favor of recently processed input. This is primarily caused by the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), a phenomenon where the gradients, which are used to update the network’s weights during training, become increasingly smaller as they are propagated back through each timestep of the sequence. As a result, the weights associated with early inputs change very little, hindering the network’s ability to learn from and remember long-term dependencies within the data.

Transformer-based models addressed these challenges and emerged as the preferred architecture for natural language processing tasks. This architecture introduced in the influential paper [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762.pdf) is a pivotal innovation in natural language processing. It forms the foundation for cutting-edge models like GPT-4, Claude, and LLaMA. The architecture was originally designed as an encoder-decoder framework. This setting uses an encoder to process input text, identifying important parts and creating a representation of the input. Meanwhile, the decoder is capable of transforming the encoder’s output, a vector of high dimensionality, back into readable text for humans. These networks can be useful in tasks such as summarization, where the decoder generates summaries conditioned based on the articles passed to the encoder. It offers additional flexibility across a wide range of tasks since the components of this architecture, the encoder, and decoder, can be used jointly or independently. Some models use the encoder part of the network to transform the text into a vector representation or use only the decoder block, which is the backbone of the Large Language Models. The next chapter will cover each of these components.

## Language Modeling

With the rise of LLMs, language modeling has become an essential part of natural language processing. It means learning the probability distribution of words within a language based on a large corpus. This learning process typically involves predicting the next token in a sequence using either classical statistical methods or novel deep learning techniques.

Large language models are trained based on the same objective to predict the next word, punctuation mark, or other elements based on the seen tokens in a text. These models become proficient by understanding the distribution of words within their training data by guessing the probability of the next word based on the context. For example, the model can complete a sentence beginning with “I live in New” with a word like “York” rather than an unrelated word such as “shoe”.

In practice, the models work with tokens, not complete words. This approach allows for more accurate predictions and text generation by more effectively capturing the complexity of human language.

## Tokenization
Tokenization is the initial phase of interacting with LLMs. It involves breaking down the input text into smaller pieces known as tokens. Tokens can range from single characters to entire words, and the size of these tokens can greatly influence the model’s performance. Some models adopt subword tokenization, breaking words into smaller segments that retain meaningful linguistic elements.

Consider the following sentence, “The child’s coloring book.”

If tokenization splits the text after every white space character. The result will be:

```python
["The", "child's", “coloring”, "book."]
```

In this approach, you’ll notice that the punctuation remains attached to the words like “child’s” and “book.”

Alternatively, tokenization can be done by separating text based on both white spaces and punctuation; the output would be:

```python
["The", "child", "'", "s", “coloring”, "book", "."]
```

The tokenization process is model-dependent. It’s important to remember that the models are released as a pair of pre-trained tokenizers and associated model weights. There are more advanced techniques, like the Byte-Pair encoding, which is used by most of the recently released models. As demonstrated in the example below, this method also divides a word such as “coloring” into two parts.

```python
["The", "child", "'", "s", “color”, “ing”, "book", "."]
```

Subword tokenization further enhances the model’s language understanding by splitting words into meaningful segments, like breaking “coloring” into “color” and “ing.” This expands the model’s vocabulary and improves its ability to grasp the nuances of language structure and morphology. Understanding that the “ing” part of a word indicates the present tense allows us to simplify how we represent words in different tenses. We no longer need to keep separate entries for the base form of a word, like “play,” and its present tense form, “playing.” By combining “play” with “ing,” we can express “playing” without needing two separate entries. This method increases the number of tokens to represent a piece of text but dramatically reduces the number of tokens we need to have in the dictionary.

The tokenization process involves scanning the entire text to identify unique tokens, which are then indexed to create a dictionary. This dictionary assigns a unique token ID to each token, enabling a standardized numerical representation of the text. When interacting with the models, this conversion of text into token IDs allows the model to efficiently process and understand the input, as it can quickly reference the dictionary to decode the meaning of each token. We will see an example of this process later in the book.

Once we have our tokens, we can process the inner workings of transformers: embeddings.

## Embeddings
The next step after tokenization is to turn these tokens into something the computer can understand and work with—this is where embeddings come into play. Embeddings are a way to translate the tokens, which are words or pieces of words, into a language of numbers that the computer can grasp. They help the model understand relationships and context. They allow the model to see connections between words and use these connections to understand text better, mainly through the attention process, as we will see.

An embedding gives each token a unique numerical ID that captures its meaning. This numerical form helps the computer see how similar two tokens are, like knowing that “happy” and “joyful” are close in meaning, even though they are different words.

This step is essential because it helps the model make sense of language in a numerical way, bridging the gap between human language and machine processing.

Initially, every token is assigned a random set of numbers as its embedding. As the model is trained—meaning as it reads and learns from lots of text—it adjusts these numbers. The goal is to tweak them so that tokens with similar meanings end up with similar sets of numbers. This adjustment is done automatically by the model as it learns from different contexts in which the tokens appear.

While the concept of numerical sets, or vectors, might sound complex, they are just a way for the model to store and process information about tokens efficiently. We use vectors because they are a straightforward method for the model to keep track of how tokens are related to each other. They are basically just large lists of numbers.


## Training/Fine-Tuning

LLMs are trained on a large corpus of text with the objective of correctly predicting the next token of a sequence. As we learned in the previous language modeling subsection, the goal is to adjust the model’s parameters to maximize the probability of a correct prediction based on the observed data. Typically, a model is trained on a huge general-purpose dataset of texts from the Internet, such as [The Pile](https://github.com/EleutherAI/the-pile) or [CommonCrawl](https://commoncrawl.org/). Sometimes, more specific datasets, such as the [StackOverflow Posts dataset](https://www.kaggle.com/datasets/stackoverflow/stackoverflow), are also an example of acquiring domain-specific knowledge. This phase is also known as the pre-training stage, indicating that the model is trained to learn language comprehension and is prepared for further tuning.

The training process adjusts the model’s weights to increase the likelihood of predicting the next token in a sequence. This adjustment is based on the training data, guiding the model towards accurate token predictions.

After pre-training, the model typically undergoes fine-tuning for a specific task. This stage requires further training on a smaller dataset for a task (e.g., text translation) or a specialized domain (e.g., biomedical, finance, etc.). Fine-tuning allows the model to adjust its previous knowledge of the specific task or domain, enhancing its performance.

The fine-tuning process can be intricate, particularly for advanced models such as GPT-4. These models employ advanced techniques and leverage large volumes of data to achieve their performance levels.

## Prediction
The model can generate text after the training or fine-tuning phase by predicting subsequent tokens in a sequence. This is achieved by inputting the sequence into the model, producing a probability distribution over the potential next tokens, essentially assigning a score to every word in the vocabulary. The next token is selected according to its score. The generation process will be repeated in a loop to predict one word at a time, so generating sequences of any length is possible. However, keeping the model’s effective context size in mind is essential.

## Context Size
The context size, or context window, is a crucial aspect of LLMs. It refers to the maximum number of tokens the model can process in a single request. Context size influences the length of text the model can handle at any one time, directly affecting the model’s performance and the outcomes it produces.

Different LLMs are designed with varying context sizes. For example, OpenAI’s “gpt-3.5-turbo-16k” model has a context window capable of handling 16,000 tokens. There is an inherent limit to the number of tokens a model can generate. Smaller models may have a capacity of up to 1,000 tokens, while larger ones like GPT-4 can manage up to 32,000 tokens as of the time we wrote this book.

## Scaling Laws
Scaling laws describe the relationship between a language model’s performance and various factors, including the number of parameters, the training dataset size, the compute budget, and the network architecture. These laws, elaborated in the [Chinchilla paper](https://arxiv.org/abs/2203.15556), provide useful insights on resource allocation for successful model training. They are also a source of many memes from the “scaling is all you need” side of the community in AI.

The following elements determine a language model’s performance:

The number of parameters (N) denotes the model’s ability to learn from data. A greater number of parameters enables the detection of more complicated patterns in data.
The size of the Training Dataset (D) and the number of tokens, ranging from small text chunks to single characters, are counted.
FLOPs (Floating Point Operations Per Second) estimate the computational resources used during training.
In their research, the authors trained the Chinchilla model, which comprises 70 billion parameters, on a dataset of 1.4 trillion tokens. This approach aligns with the scaling law proposed in the paper: for a model with X parameters, the optimal training involves approximately X * 20 tokens. For example, a model with 100 billion parameters would ideally be trained on about 2 trillion tokens.

With this approach, despite its smaller size compared to other LLMs, the Chinchilla model outperformed them all. It improved language modeling and task-specific performance using less memory and computational power. Find the paper [“Training Compute-Optimal Large Language Models.”](https://arxiv.org/abs/2203.15556) at [towardsai.net/book.](https://towardsai.net/book)

## Emergent Abilities in LLMs

Emergent abilities in LLMs describe the phenomena in which new skills emerge unexpectedly as model size grows. These abilities, including arithmetic, answering questions, summarizing material, and others, are not explicitly taught to the model throughout its training. Instead, they emerge spontaneously when the model’s scaling increases, hence the word “emergent.”

LLMs are probabilistic models that learn natural language patterns. When these models are ramped up, their pattern recognition capacity improves quantitatively while also changing qualitatively.

Traditionally, models required task-specific fine-tuning and architectural adjustments to execute specific tasks. However, scaled-up models can perform these jobs without architectural changes or specialized tuning. They accomplish this by interpreting tasks using natural language processing. LLMs’ ability to accomplish various functions without explicit fine-tuning is a significant milestone.

What’s more remarkable is how these abilities show themselves. LLMs swiftly and unpredictably progress from near-zero to sometimes state-of-the-art performance as their size grows. This phenomenon indicates that these abilities arise from the model’s scale rather than being clearly programmed into the model.

This growth in model size and the expansion of training datasets, accompanied by substantial increases in computational costs, paved the way for the emergence of today’s Large Language Models. Examples of such models include Cohere Command, GPT-4, and LLaMA, each representing significant milestones in the evolution of language modeling.

## Prompts

The text (or images, numbers, tables…) we provide to LLMs as instructions is commonly called prompts. Prompts are instructions given to AI systems like OpenAI’s GPT-3 and GPT-4, providing context to generate human-like text—the more detailed the prompt, the better the model’s output.

Concise, descriptive, and short (depending on the task) prompts generally lead to more effective results, allowing for the LLM’s creativity while guiding it toward the desired output. Using specific words or phrases can help focus the model on generating relevant content. Creating effective prompts requires a clear purpose, keeping things simple, strategically using keywords, and assuring actionability. Testing prompts before final use is critical to ensure the output is relevant and error-free. Here are some prompting tips:

1. Use Precise Language: Precision in your prompt can significantly improve the accuracy of the output.
   - Less Precise: “Write about dog food.”
   - More Precise: “Write a 500-word informative article about the dietary needs of adult Golden Retrievers.”

2. Provide Sufficient Context: Context helps the model understand the expected output:
    - Less Contextual: “Write a story.”
    - More Contextual: “Write a short story set in Victorian England featuring a young detective solving his first major case.”

3. Test Variations: Experiment with different prompt styles to find the most effective approach:
    - Initial: “Write a blog post about the benefits of yoga.”
    - Variation 1: “Compose a 1000-word blog post detailing the physical and mental benefits of regular yoga practice.”
    - Variation 2: “Create an engaging blog post that highlights the top 10 benefits of incorporating yoga into a daily routine.”

4. Review Outputs: Always double-check automated outputs for accuracy and relevance before publishing.
    - Before Review: “Yoga is a great way to improve your flexibility and strength. It can also help reduce stress and improve mental clarity. However, it’s important to remember that all yoga poses are suitable for everyone.”
    - After Review (corrected): “Yoga is a great way to improve your flexibility and strength. It can also help reduce stress and improve mental clarity. However, it’s important to remember that not all yoga poses are suitable for everyone. Always consult with a healthcare professional before starting any new exercise regimen.”

## Hallucinations and Biases in LLMs
Hallucinations in AI systems refer to instances where these systems produce outputs, such as text or visuals, inconsistent with facts or the available inputs. One example would be if ChatGPT provides a compelling but factually wrong response to a question. These hallucinations show a mismatch between the AI’s output and real-world knowledge or context.

In LLMs, hallucinations occur when the model creates outputs that do not correspond to real-world facts or context. This can lead to the spread of disinformation, especially in crucial industries like healthcare and education, where information accuracy is critical. Bias in LLMs can also result in outcomes that favor particular perspectives over others, possibly reinforcing harmful stereotypes and discrimination.

An example of a hallucination could be if a user asks, “Who won the World Series in 2025?” and the LLM responds with a specific winner. As of the current date (Jan 2024), the event has yet to occur, making any response speculative and incorrect.

Additionally, Bias in AI and LLMs is another critical issue. It refers to these models’ inclination to favor specific outputs or decisions based on their training data. If the training data primarily originates from a particular region, the model may be biased toward that region’s language, culture, or viewpoints. In cases where the training data encompasses biases, like gender or race, the resulting outputs from the AI system could be biased or discriminatory.

For example, if a user asks an LLM, “Who is a nurse?” and it responds, “She is a healthcare professional who cares for patients in a hospital,” this demonstrates a gender bias. The paradigm inherently associates nursing with women, which needs to adequately reflect the reality that both men and women can be nurses.

Mitigating hallucinations and bias in AI systems involves refining model training, using verification techniques, and ensuring the training data is diverse and representative. Finding a balance between maximizing the model’s potential and avoiding these issues remains challenging.

Amazingly, these “hallucinations” might be advantageous in creative fields such as fiction writing, allowing for the creation of new and novel content. The ultimate goal is to create powerful, efficient but also trustworthy, fair, and reliable LLMs. We can maximize the promise of LLMs while minimizing their hazards, ensuring that the advantages of this technology are available to all.

Translation with LLMs (GPT-3.5 API)
Now, we can combine all we have learned to demonstrate how to interact with OpenAI’s proprietary LLM through their API, instructing the model to perform translation. To generate text using LLMs like those provided by OpenAI, you first need an API key for your Python environment. Here’s a step-by-step guide to generating this key:

Create and log into your OpenAI account.
After logging in, select ‘Personal’ from the top-right menu and click “View API keys.”
You’ll find the “Create new secret key” button on the API keys page. Click on it to generate a new secret key. Remember to save this key securely, as it will be used later.
After generating your API key, you can securely store it in a .env file using the following format:

```python
OPENAI_API_KEY="<YOUR-OPENAI-API-KEY>"
```

Every time you initiate a Python script including the following lines, your API key will be automatically loaded into an environment variable named OPENAI_API_KEY. The openai library subsequently uses this variable for text generation tasks. The .env file must be in the same directory as the Python script.

```python
from dotenv import load_dotenv

load_dotenv()
```
Now, the model is ready for interaction! Here’s an example of using the model for a language translation from English to French. The code below sends the prompt as a message with a user role, using the OpenAI Python package to send and retrieve requests from the API. There is no need for concern if you do not understand all the details, as we will use the OpenAI API more thoroughly in Chapter 5. It would be best if you focused on the messages argument for now, which receives the prompt that directs the model to execute the translation task.

```python
from dotenv import load_dotenv
load_dotenv()
import os
import openai

# English text to translate
english_text = "Hello, how are you?"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f'''Translate the following English text to French: "{english_text}"'''}
  ],
)

print(response['choices'][0]['message']['content'])
```

Bonjour, comment ça va?

💡 You can safely store sensitive information, such as API keys, in a separate file with dotenv and avoid accidentally exposing it in your code. This is especially important when working with open-source projects or sharing your code with others, as it ensures the security of sensitive information.

## Control LLMs Output by Providing Examples
Few-shot learning, which is one of the emergent abilities of LLMs, means providing the model with a small number of examples before making predictions. These examples serve a dual purpose: they “teach” the model in its reasoning process and act as “filters,” aiding the model in identifying relevant patterns within its dataset. Few-shot learning allows for the adaptation of the model to new tasks. While LLMs like GPT-3 show proficiency in language modeling tasks such as machine translation, their performance can vary on tasks that require more complex reasoning.

In few-shot learning, the examples presented to the model help discover relevant patterns in the dataset. The datasets are effectively encoded into the model’s weights during the training, so the model looks for patterns that significantly connect with the provided samples and uses them to generate its output. As a result, the model’s precision improves by adding more examples, allowing for a more targeted and relevant response.

Here is an example of few-shot learning, where we provide examples through different message types on how to describe movies with emojis to the model. (We will cover the different message types later in the book.) For instance, the movie “Titanic” might be presented using emojis for a cruise ship, waves, a heart, etc., or how to represent “The Matrix” movie. The model picks up on these patterns and manages to accurately describe the movie “Toy Story” using emojis of toys.

```python
from dotenv import load_dotenv
load_dotenv()
import os
import openai

# Prompt for summarization
prompt = """
Describe the following movie using emojis.

{movie}: """

examples = [
    { "input": "Titanic", "output": "🛳️🌊❤️🧊🎶🔥🚢💔👫💑" },
    { "input": "The Matrix", "output": "🕶️💊💥👾🔮🌃👨🏻‍💻🔁🔓💪" }
]

movie = "Toy Story"
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(movie=examples[0]["input"])},
        {"role": "assistant", "content": examples[0]["output"]},
        {"role": "user", "content": prompt.format(movie=examples[1]["input"])},
        {"role": "assistant", "content": examples[1]["output"]},
        {"role": "user", "content": prompt.format(movie=movie)},
  ]
)

print(response['choices'][0]['message']['content'])

🧸🤠👦🧒🎢🌈🌟👫🚁👽🐶🚀
```

It’s fascinating how the model, with just two examples, can identify a complex pattern, such as associating a film title with a sequence of emojis. This ability is achievable only with a model that possesses an in-depth understanding of the film’s story and the meaning of the emojis, allowing it to merge the two and respond to inquiries based on its own interpretation. 

## From Language Models to Large Language Models
The evolution of language models has seen a paradigm shift from pre-trained language models (LMs) to the creation of Large Language Models (LLMs). LMs, such as ELMo and BERT, first captured context-aware word representations through pre-training and fine-tuning for specific tasks. However, the introduction of LLMs, as demonstrated by GPT-3 and PaLM, proved that scaling model size and data can unlock emergent skills that outperform their smaller counterparts. Through in-context learning, these LLMs can handle more complex tasks.

## Emergent Abilities in LLMs
As we discussed, an ability is considered emergent when larger models exhibit it, but it’s absent in smaller models—a key factor contributing to the success of Large Language Models. Emergent abilities in Large Language Models (LLMs) are empirical phenomena that occur when the size of language models exceeds specific thresholds. As we increase the models’ size, emergent abilities become more evident, influenced by aspects like the computational power used in training and the model’s parameters.

## What Are Emergent Abilities
This phenomenon indicates that the models are learning and generalizing beyond their pre-training in ways that were not explicitly programmed or anticipated. A distinct pattern emerges when these abilities are depicted on a scaling curve. Initially, the model’s performance appears almost random, but it significantly improves once a certain scale threshold is reached. This phenomenon is known as a phase transition, representing a dramatic behavior change that would not have been apparent from examining smaller-scale systems.

Scaling language models have predominantly focused on increasing the amount of computation, expanding the model parameters, and enlarging the training dataset size. New abilities can sometimes emerge with reduced training computation or fewer model parameters, especially when models are trained on higher-quality data. Additionally, the appearance of emergent abilities is influenced by factors such as the volume and quality of the data and the quantity of the model’s parameters. Emergent abilities in Large Language Models surface as the models are scaled up and are not predictable by merely extending the trends observed in smaller models.

# Evaluation Benchmarks for Emergent Abilities
Several benchmarks are used to evaluate the emergent abilities of language models, such as BIG-Bench, TruthfulQA, the Massive Multi-task Language Understanding (MMLU) benchmark, and the Word in Context (WiC) benchmark. Key benchmarks include:

1. [BIG-Bench suite](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/README.md) comprises over 200 benchmarks testing a wide array of tasks, such as arithmetic operations (example: “Q: What is 132 plus 762? A: 894), transliteration from the International Phonetic Alphabet (IPA), and word unscrambling. These tasks assess a model’s capacity to perform calculations, manipulate and use rare words, and work with alphabets. (example: “English: The 1931 Malay census was an alarm bell. IPA: ðə 1931 ˈmeɪleɪ ˈsɛnsəs wɑz ən əˈlɑrm bɛl.”) The performance of models like GPT-3 and LaMDA on these tasks usually starts near zero but shows a significant increase above random at a certain scale, indicative of emergent abilities. More details on these benchmarks can be found in the Github repository.

2. [TruthfulQA benchmark](https://github.com/sylinrl/TruthfulQA) evaluates a model’s ability to provide truthful responses. It includes two tasks: generation, where the model answers a question in one or two sentences, and multiple-choice, where the model selects the correct answer from four options or True/False statements. As the Gopher model is scaled to its largest size, its performance improves significantly, exceeding random outcomes by over 20%, which signifies the emergence of this ability.

3. [Massive Multi-task Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) assesses a model’s world knowledge and problem-solving skills across 57 diverse tasks, including elementary mathematics, US history, and computer science. While GPTs, Gopher, and Chinchilla models of a certain scale do not outperform random guessing on average across all topics, a larger size model shows improved performance, suggesting the emergence of this ability.

4. [The Word in Context (WiC)](https://arxiv.org/abs/1808.09121v3) benchmark focuses on semantic understanding and involves a binary classification task for context-sensitive word embeddings. It requires determining if target words (verbs or nouns) in two contexts share the same meaning. Models like Chinchilla initially fail to surpass random performance in one-shot tasks, even at large scales. However, when models like PaLM are scaled to a much larger size, above-random performance emerges, indicating the emergence of this ability at a larger scale.

## Factors Leading To Emergent Abilities
 

- Multi-step reasoning involves instructing a model to perform a series of intermediate steps before providing the final result. This approach, known as chain-of-thought prompting, becomes more effective than standard prompting only when applied to sufficiently large models.
- Another strategy is fine-tuning a model on various tasks presented as Instruction Following. This method shows improved performance only with models of a certain size, underlining the significance of scale in achieving advanced capabilities.

Risks With Emergent Abilities
As language models are scaled up, emergent risks also become a concern. These include societal challenges related to accuracy, bias, and toxicity. Adopting strategies that encourage models to be “helpful, harmless, and honest” can mitigate these risks.

For instance, the [WinoGender benchmark](https://uclanlp.github.io/corefBias/overview), which assesses gender bias in occupational contexts, has shown that while scaling can enhance model performance, it may also amplify biases, especially in ambiguous situations. Larger models tend to memorize training data more, but methods like deduplication can reduce this risk.

Other risks involve potential vulnerabilities or harmful content synthesis that might be more prevalent in future language models or remain uncharacterized in current models.

## A Shift Towards General-Purpose Models
The emergence of new abilities has shifted the NLP community’s perspective and utilization of these models. While NLP traditionally focused on task-specific models, the scaling of models has spurred research on “general-purpose” models capable of handling a wide range of tasks not explicitly included in their training.

This shift is evident in instances where scaled, few-shot prompted general-purpose models have outperformed task-specific models that were fine-tuned. Examples include GPT-3 setting new benchmarks in TriviaQA and PiQA, PaLM excelling in arithmetic reasoning, and the multimodal Flamingo model achieving top performance in visual question answering. Furthermore, the ability of general-purpose models to execute tasks with minimal examples has expanded their applications beyond traditional NLP research. These include translating natural language instructions for robotic execution, user interaction, and multi-modal reasoning.

## Expanding the Context Window

### The Importance of Context Length
Context window in language models represents the number of input tokens the model can process simultaneously. In models like [GPT-4](https://openai.com/research/gpt-4), it currently stands at approximately 32K or roughly 50 pages of text. However, recent advancements have extended this to an impressive 100K tokens or about 156 pages, as seen in [Claude by Anthropic](https://www.anthropic.com/index/100k-context-windows).

Context length primarily enables the model to process and comprehend larger datasets simultaneously, offering a deeper understanding of the context. This feature is particularly beneficial when inputting a substantial amount of specific data into a language model and posing questions related to this data. For example, when analyzing a lengthy document about a particular company or issue, a larger context window allows the language model to review and remember more of this unique information, resulting in more accurate and tailored responses.

## Limitations of the Original Transformer Architecture
Despite its strengths, the original transformer architecture faces challenges in handling extensive context lengths. Specifically, the attention layer operations in the transformer have quadratic time and space complexity (represented with ) in relation to the number of input tokens, . As the context length expands, the computational resources required for training and inference increase substantially.

To better understand this, let’s examine the computational complexity of the transformer architecture. The complexity of the attention layer in the transformer model is , where is the context length (number of input tokens) and is the embedding size.

This complexity stems from two primary operations in the attention layer: linear projections to create Query, Key, and Value matrices (complexity ~ ) and the multiplication of these matrices (complexity ~ ). As the context length or embedding size increases, the computational complexity also grows quadratically, presenting a challenge for processing larger context lengths.

## Optimization Techniques to Expand the Context Window
Despite the computational challenges associated with the original transformer architecture, researchers have developed a range of optimization techniques to enhance the transformer’s efficiency and increase its context length capacity to 100K tokens:

1. [ALiBi Positional Encoding:](https://arxiv.org/abs/2108.12409) The original transformer used Positional Sinusoidal Encoding, which has trouble inferring larger context lengths. On the other hand, ALiBi (Attention with Linear Biases) is a more scalable solution. This positional encoding technique allows the model to be trained in smaller contexts and then fine-tuned in bigger contexts, making it more adaptive to different context sizes.

2. [Sparse Attention:](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html) Sparse Attention addresses the computational challenge by focusing attention scores on a subset of tokens. This method significantly decreases the computing complexity to a linear scale with respect to the number of tokens n, resulting in a significant reduction in overall computational demand.

3. [FlashAttention:](https://arxiv.org/abs/2205.14135) FlashAttention restructures the attention layer calculation for GPU efficiency. It divides input matrices into blocks and then processes attention output with reference to these blocks, optimizing GPU memory utilization and increasing processing efficiency.

4. [Multi-Query Attention (MQA):](https://arxiv.org/pdf/1911.02150.pdf) MQA reduces memory consumption in the key/value decoder cache by aggregating weights across all attention heads during linear projection of the Key and Value matrices. This consolidation results in more effective memory utilization.

## FlashAttention-2
[FlashAttention-2](https://crfm.stanford.edu/2023/07/17/flash2.html?utm_content=bufferca8a7&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer) emerges as an advancement over the original FlashAttention, focusing on optimizing the speed and memory efficiency of the attention layer in transformer models. This upgraded version is redeveloped from the ground up utilizing Nvidia’s new primitives. It performs approximately 2x faster than its predecessor, achieving up to 230 TFLOPs on A100 GPUs.

FlashAttention-2 improves on the original FlashAttention in various ways.

- Changing the algorithm to spend more time on matmul FLOPs minimizes the quantity of non-matmul FLOPs, which are 16x more expensive than matmul FLOPs.
- It optimizes parallelism across batch size, headcount, and sequence length dimensions, leading to significant acceleration, particularly for long sequences.
- It enhances task partitioning within each thread block to reduce synchronization and communication between warps, resulting in fewer shared memory reads/writes.
- It adds features such as support for attention head dimensions up to 256 and multi-query attention (MQA), further expanding the context window.
With these enhancements, FlashAttention-2 is a successful step toward context window expansion (while still retaining the underlying restrictions of the original transformer architecture).

## LongNet: A Leap Towards Billion-Token Context Window
LongNet represents a transformative advancement in the field of transformer optimization, as detailed in the paper [“LONGNET: Scaling Transformers to 1,000,000,000 Tokens”](https://arxiv.org/pdf/2307.02486.pdf). This innovative approach is set to extend the context window of language models to an unprecedented 1 billion tokens, significantly enhancing their ability to process and analyze large volumes of data.

The primary advancement in LongNet is the implementation of “dilated attention.” This innovative attention mechanism allows for an exponential increase in the attention field as the gap between tokens widens, inversely reducing attention calculations as the distance between tokens increases. (since every token will attend to a smaller number of tokens). This design approach balances the limited attention resources and the need to access every token in the sequence.

LongNet’s dilated attention mechanism has a linear computational complexity, a major improvement over the normal transformer’s quadratic difficulty.

# A Timeline of the Most Popular LLMs
Here’s the timeline of some of the most popular LLMs in the last five years.

- [2018 GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
 Introduced by OpenAI, GPT-1 laid the foundation for the GPT series with its generative, decoder-only transformer architecture. It pioneered the combination of unsupervised pretraining and supervised fine-tuning for natural language text prediction.
- [2019 GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
 Building on GPT-1’s architecture, GPT-2 expanded the model size to 1.5 billion parameters, demonstrating the model’s versatility across a range of tasks using a unified format for input, output, and task information.
- [2020 GPT-3](https://en.wikipedia.org/wiki/OpenAI_Codex)
 Released in 2020, GPT-3 marked a substantial leap with 175 billion parameters, introducing in-context learning (ICL). This model showcased exceptional performance in various NLP tasks, including reasoning and domain adaptation, highlighting the potential of scaling up model size.
- [2021 Codex](https://blog.google/technology/ai/lamda/)
 OpenAI introduced Codex in July 2021. It is a GPT-3 variant fine-tuned on a corpus of GitHub code and exhibited advanced programming and mathematical problem-solving capabilities, demonstrating the potential of specialized training.
- [2021 LaMDA](https://blog.google/technology/ai/lamda/)
 Researchers from DeepMind introduced LaMDA (Language Models for Dialog Applications). LaMDA focused on dialog applications, boasting 137 billion parameters. It aimed to enhance dialog generation and conversational AI.
- [2021 Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval)
 In 2021, DeepMind’s Gopher, with 280 billion parameters, approached human-level performance on the MMLU benchmark but faced challenges like biases and misinformation.
- [2022 InstructGPT](https://arxiv.org/abs/2203.02155)
 In 2022, InstructGPT, an enhancement to GPT-3, utilized reinforcement learning from human feedback to improve instruction-following and content safety, aligning better with human preferences
- [2022 Chinchilla](https://arxiv.org/abs/2203.15556)
DeepMind’s Chinchilla introduced in 2022, with 70 billion parameters, optimized compute resource usage based on scaling laws, achieving significant accuracy improvements on benchmarks.
- [2022 PaLM](https://arxiv.org/abs/2204.02311)
 Pathways Language Model (PaLM) was introduced by Google Research in 2022. Google’s PaLM, with an astounding 540 billion parameters, demonstrated exceptional few-shot performance, benefiting from Google’s Pathways system for distributed computation.
- [2022 ChatGPT](https://openai.com/blog/chatgpt)
 In November 2022, OpenAI’s ChatGPT, based on GPT-3.5 and GPT-4, was tailored for conversational AI and showed proficiency in human-like communication and reasoning.
- [2023 LLaMA](https://arxiv.org/abs/2302.13971)
 Meta AI developed LLaMA (Large Language Model Meta AI) in February 2023. It introduced a family of massive language models with parameters ranging from 7 billion to 65 billion. The publication of LLaMA broke the tradition of limited access by making its model weights available to the scientific community under a noncommercial license. Subsequent innovations, such as LLaMA 2 and other chat formats, stressed accessibility even further, this time with a commercial license.
- [2023 GPT-4](https://arxiv.org/abs/2303.08774)
 In March 2023, GPT-4 expanded its capabilities to multimodal inputs, outperforming its predecessors in various tasks and representing another significant step in LLM development.
- [2024 Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/)
 Gemini 1.5 (from Google) features a significant upgrade compared to the previous iteration of the model with a new Mixture-of-Experts architecture and multimodal model capability, Gemini 1.5 Pro, which supports advanced long-context understanding and a context window of up to 1 million tokens. The context window size is larger than any other model available today. The model is accessible through Google’s proprietary API.
- [2024 Gemma](https://blog.google/technology/developers/gemma-open-models/)
 Google has also released the Gemma model in two versions: 2 billion and 7 billion parameters. These models were developed during the training phase that produced the Gemini model and are now publicly accessible. Users can access these models in both pre-trained and instruction-tuned formats.
- [2024 Claude 3 Opus](https://www.anthropic.com/news/claude-3-family)
 The newest model from Anthropic, the Claude 3 Opus, is available via their proprietary API. It is one of the first models to achieve scores comparable to or surpassing GPT-4 across different benchmarks. With a context window of 200K tokens, it is advertised for its exceptional recall capabilities, regardless of the position of the information within the window.
- [2024 Mistral](https://arxiv.org/abs/2401.04088)
 Following their publication detailing the Mixture of Experts architecture, they have now made the 8x22 billion base model available to the public. This model is the best open-source option currently accessible for use. Despite this, it still does not outperform the performance of closed-source models like GPT-4 or Claude.
- [2024 Infinite Attention](https://arxiv.org/abs/2404.07143)
 Google’s recent paper, speculated to be the base of the Gemini 1.5 Pro model, explores techniques that could indefinitely expand the model’s context window size. Speculation surfaced because the paper released alongside the Gemini model mentioned that the model could perform exceptionally well with up to 10 million tokens. However, a model with these specifications has yet to be released. This approach is described as a plug-and-play solution that can significantly enhance any model’s few-shot learning performance without context size constraints.
 
 If you want to dive deeper into these models, we suggest reading the paper [“A Survey of Large Language Models”](https://arxiv.org/pdf/2303.18223.pdf).

# History of NLP/LLMs
This is a journey through the growth of language modeling models, from early statistical models to the birth of the first Large Language Models (LLMs). Rather than an in-depth technical study, this chapter presents a story-like exploration of model building. Don’t worry if certain model specifics appear complicated.

## The Evolution of Language Modeling
The evolution of natural language processing (NLP) models is a story of constant invention and improvement. The Bag of Words model, a simple approach for counting word occurrences in documents, began in 1954. Then, in 1972, TF-IDF appeared, improving on this strategy by altering word counts based on rarity or frequency. The introduction of Word2Vec in 2013 marked a significant breakthrough. This model used word embeddings to capture subtle semantic links between words that previous models could not.

Following that, Recurrent Neural Networks (RNNs) were introduced. RNNs were adept at learning patterns in sequences, allowing them to handle documents of varied lengths effectively.

The launch of the transformer architecture in 2017 signified a paradigm change in the area. During output creation, the model’s attention mechanism allowed it to focus on the most relevant elements of the input selectively. This breakthrough paved the way for BERT in 2018. BERT used a bidirectional transformer, significantly increasing performance in various traditional NLP workloads.

The years that followed saw a rise in model developments. Each new model, such as RoBERTa, XLM, ALBERT, and ELECTRA, introduced additional enhancements and optimizations, pushing the bounds of what was feasible in NLP.

## Model’s Timeline

- [1954 Bag of Words (BOW)](https://en.wikipedia.org/wiki/Bag-of-words_model)
 The Bag of Words model was a basic approach that tallied word occurrences in manuscripts. Despite its simplicity, it could not consider word order or context.
- [1972 TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
 TF-IDF expanded on BOW by giving more weight to rare words and less to common terms, improving the model’s ability to detect document relevancy. Nonetheless, it made no mention of word context.
- [2013 Word2Vec](https://arxiv.org/abs/1301.3781)
 Word embeddings are high-dimensional vectors encapsulating semantic associations, as described by Word2Vec. This was a substantial advancement in capturing textual semantics.
- [2014 RNNs in Encoder-Decoder architectures](https://en.wikipedia.org/wiki/Recurrent_neural_network)
 RNNs were a significant advancement, capable of computing document embeddings and adding word context. They grew to include LSTM (1997) for long-term dependencies and Bidirectional RNN (1997) for context understanding. Encoder-Decoder RNNs (2014) improved on this method.
- [2017 Transformer](https://arxiv.org/abs/1706.03762)
 The transformer, with its attention mechanisms, greatly improved embedding computation and alignment between input and output, revolutionizing NLP tasks.
- [2018 BERT](https://arxiv.org/abs/1810.04805)
 BERT, a bidirectional transformer, achieved impressive NLP results using global attention and combined training objectives.
- [2018 GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
 The transformer architecture was used to create the first autoregressive model, GPT. It then evolved into [GPT-2 2019](), a larger and more optimized version of GPT pre-trained on WebText, and [GPT-3 2020](), a larger and more optimized version of GPT-2 pre-trained on Common Crawl.
- [2019 CTRL](https://arxiv.org/abs/1909.05858)
 CTRL, similar to GPT, introduced control codes enabling conditional text generation. This feature enhanced control over the content and style of the generated text.
- [2019 Transformer-XL](https://arxiv.org/abs/1901.02860)
 Transformer-XL innovated by reusing previously computed hidden states, allowing the model to maintain a longer contextual memory. This enhancement significantly improved the model’s ability to handle extended text sequences.
- [2019 ALBERT](https://arxiv.org/abs/1909.11942)
 ALBERT offered a more efficient version of BERT by implementing Sentence Order Prediction instead of Next Sentence Prediction and employing parameter-reduction techniques. These changes resulted in lower memory usage and expedited training.
- [2019 RoBERTa](https://arxiv.org/abs/1907.11692)
 RoBERTa improved upon BERT by introducing dynamic Masked Language Modeling, omitting the Next Sentence Prediction, using the BPE tokenizer, and employing better hyperparameters for enhanced performance.
- [2019 XLM](https://arxiv.org/abs/1901.07291)
 XLM was a multilingual transformer, pre-trained using a variety of objectives, including Causal Language Modeling, Masked Language Modeling, and Translation Language Modeling, catering to multilingual NLP tasks.
- [2019 XLNet](https://arxiv.org/abs/1906.08237)
 XLNet combined the strengths of Transformer-XL with a generalized autoregressive pretraining approach, enabling the learning of bidirectional dependencies and offering improved performance over traditional unidirectional models.
- [2019 PEGASUS](https://arxiv.org/abs/1912.08777)
PEGASUS featured a bidirectional encoder and a left-to-right decoder, pre-trained using objectives like Masked Language Modeling and Gap Sentence Generation, optimizing it for summarization tasks.
- [2019 DistilBERT](https://arxiv.org/abs/1910.01108)
 DistilBERT presented a smaller, faster version of BERT, retaining over 95% of its performance. This model was trained using distillation techniques to compress the pre-trained BERT model.
- [2019 XLM-RoBERTa](https://arxiv.org/pdf/1911.02116.pdf)
 XLM-RoBERTa was a multilingual adaptation of RoBERTa, trained on a diverse multilanguage corpus, primarily using the Masked Language Modeling objective, enhancing its multilingual capabilities.
- [2019 BART](https://arxiv.org/abs/1910.13461)
 BART, with a bidirectional encoder and a left-to-right decoder, was trained by intentionally corrupting text and then learning to reconstruct the original, making it practical for a range of generation and comprehension tasks.
- [2019 ConvBERT](https://arxiv.org/abs/2008.02496)
 ConvBERT innovated by replacing traditional self-attention blocks with modules incorporating convolutions, allowing for more effective handling of global and local contexts within the text.
- [2020 Funnel Transformer](https://arxiv.org/abs/2006.03236)
 Funnel Transformer innovated by progressively compressing the sequence of hidden states into a shorter sequence, effectively reducing computational costs while maintaining performance.
- [2020 Reformer](https://arxiv.org/abs/2001.04451)
 Reformer offered a more efficient version of the transformer. It utilized locality-sensitive hashing for attention mechanisms and axial position encoding, among other optimizations, to enhance efficiency.
- [2020 T5](https://arxiv.org/abs/1910.10683)
 T5 approached NLP tasks as a text-to-text problem. It was trained using a mixture of unsupervised and supervised tasks, making it versatile for various applications.
- [2020 Longformer](https://arxiv.org/abs/2004.05150)
 Longformer adapted the transformer architecture for longer documents. It replaced traditional attention matrices with sparse versions, improving training efficiency and better handling of longer texts.
- [2020 ProphetNet](https://arxiv.org/abs/2001.04063)
 ProphetNet was trained using a Future N-gram Prediction objective, incorporating a unique self-attention mechanism. This model aimed to improve sequence-to-sequence tasks like summarization and question-answering.
- [2020 ELECTRA](https://arxiv.org/abs/2003.10555)
 ELECTRA presented a novel approach, trained with a Replaced Token Detection objective. It offered improvements over BERT in efficiency and performance across various NLP tasks.
- [2021 Switch Transformers](https://arxiv.org/abs/2101.03961)
 Switch Transformers introduced a sparsely-activated expert model, a new spin on the Mixture of Experts (MoE) approach. This design allowed the model to manage a broader array of tasks more efficiently, marking a significant step towards scaling up transformer models.

## Recap
The advancements in natural language processing, beginning with the essential Bag of Words model, led us to the advanced and highly sophisticated transformer-based models we have today. Large language models (LLMs) are powerful architectures trained on massive amounts of text data that can comprehend and generate writing that nearly resembles human language. Built on transformer designs, they excel at capturing long-term dependencies in language and producing text via an auto-regressive process.

The years 2020 and 2021 were key moments in the advancement of Large Language Models (LLMs). Before this, language models’ primary goal was to generate coherent and contextually suitable messages. However, advances in LLMs throughout these years resulted in a paradigm shift.

The journey from pre-trained language models to Large Language Models (LLMs) is marked by distinctive features of LLMs, such as the impact of scaling laws and the emergence of abilities like in-context learning, step-by-step reasoning techniques, and instruction following. These emergent abilities are central to the success of LLMs, showcased in scenarios like few-shots and augmented prompting. However, scaling also brings challenges like bias and toxicity, necessitating careful consideration.

Emergent abilities in LLMs have shifted the focus towards general-purpose models, opening up new applications beyond traditional NLP research. The expansion of context windows also played a key role in this shift. Innovations like FlashAttention-2, which optimizes the attention layer’s speed and memory utilization, and LongNet, which introduced the “dilated attention” method, have paved the way for context windows to potentially grow to 1 billion tokens.

In this chapter, we explored the fundamentals of LLMs, their history, and evolution. We experimented with concepts such as tokenization, context, and few-shot learning with practical examples and identified the inherent problems in LLMs, such as hallucinations and biases, emphasizing mitigation.

💡 Research papers on evaluation benchmarks and optimization techniques are available at [towardsai.net/book](http://towardsai.net/book).

# Chapter II: LLM Architectures and Landscape

## Understanding Transformers
The transformer architecture has demonstrated its versatility in various applications. The original network was presented as an encoder-decoder architecture for translation tasks. The next evolution of transformer architecture began with the introduction of encoder-only models like BERT, followed by the introduction of decoder-only networks in the first iteration of GPT models.

The differences extend beyond just network design and also encompass the learning objectives. These contrasting learning objectives play a crucial role in shaping the model’s behavior and outcomes. Understanding these differences is essential for selecting the most suitable architecture for a given task and achieving optimal performance in various applications.

In this chapter, we will explore transformers in more depth, providing a comprehensive understanding of their various components and the network’s inner mechanisms. We will also look into the seminal paper “Attention is all you need”.

We will also load pre-trained models to highlight the distinctions between transformer and GPT architectures and examine the latest innovations in the field with large multimodal models (LMMs).

## Attention Is All You Need
It is a highly memorable title in the field of natural language processing (NLP). The paper “Attention is All You Need” marked a significant milestone in developing neural network architectures for NLP. This collaborative effort between Google Brain and the University of Toronto introduced the transformer, an encoder-decoder network harnessing attention mechanisms for automatic translation tasks. The transformer model achieved a new state-of-the-art score of 41.8 on the (WMT 2014 dataset) English-to-French translation task. Remarkably, this level of performance was achieved after just 3.5 days of training on eight GPUs, showcasing a drastic reduction in training costs compared to previous models.

Transformers have drastically changed the field and have demonstrated remarkable effectiveness across different tasks beyond translation, including classification, summarization, and language generation. A key innovation of the transformer is its highly parallelized network structure, which enhances both efficiency and effectiveness in training.

## The Architecture
Now, let’s examine the essential components of a transformer model in more detail. As displayed in the diagram below, the original architecture was designed for sequence-to-sequence tasks (where a sequence is inputted and an output is generated based on it), such as translation. In this process, the encoder creates a representation of the input phrase, and the decoder generates its output using this representation as a reference.

Encoder-Decoder Image

The overview of Transformer architecture. The left component is called the encoder, connected to the decoder using a cross-attention mechanism.

Further research into architecture resulted in its division into three unique categories, distinguished by their versatility and specialized capabilities in handling different tasks.

- The encoder-only category is dedicated to extracting context-aware representations from input data. A representative model from this category is BERT, which can be useful for classification tasks.
- The encoder-decoder category facilitates sequence-to-sequence tasks such as translation, summarization and training multimodal models like caption generators. An example of a model under this classification is BART.
- The decoder-only category is specifically designed to produce outputs by following the instructions provided, as demonstrated in LLMs. A representative model in this category is the GPT family.
Next, we will cover the contrasts between these design choices and their effects on different tasks. However, as you can see from the diagram, several building blocks, like embedding layers and the attention mechanism, are shared on both the encoder and decoder components. Understanding these elements will help improve your understanding of how the models operate internally. This section outlines the key components and then demonstrates how to load an open-source model to trace each step.

## Input Embedding
As we’ve seen in the transformer architecture, the initial step is to turn input tokens (words or subwords) into embeddings. These embeddings are high-dimensional vectors that capture the semantic features of the input tokens. You can see them as a large list of characteristics representing the words being embedded. This list contains thousands of numbers that the model learns by itself to represent our world. Instead of working with sentences, words, and synonyms to compare things together, requiring an understanding of our language, it works with these lists of numbers to compare them numerically with basic calculations, subtracting and adding those vectors together to see if they are similar or not. It looks much more complex than understanding words themselves, doesn’t it? This is why the size of these embedding vectors is pretty large. When you cannot understand meanings and words, you need thousands of values representing them. This size varies depending on the model’s architecture. GPT-3 by OpenAI, for example, employs 12,000-dimensional embedding vectors, but smaller models such as BERT employ 768-dimensional embeddings. This layer enables the model to understand and process the inputs effectively, serving as the foundation for all subsequent layers.

## Positional Encoding
Earlier models, such as Recurrent Neural Networks (RNNs), processed inputs sequentially, one token at a time, naturally preserving the text’s order. Unlike these, transformers do not have built-in sequential processing capabilities. Instead, they employ positional encodings to maintain the order of words within a phrase for the next layers. These encodings are vectors filled with unique values at each index, which, when combined with input embeddings, provide the model with data regarding the tokens’ relative or absolute positions within the sequence. These vectors encode each word’s position, ensuring that the model identifies word order, which is essential for interpreting the context and meaning of a sentence.

## Self-Attention Mechanism
The self-attention mechanism is at the heart of the transformer model, calculating a weighted total of the embeddings of all words in a phrase. These weights are calculated using learned “attention” scores between words. Higher “attention” weights will be assigned to terms that are more relevant to one another. Based on the inputs, this is implemented using Query, Key, and Value vectors. Here is a brief description of each vector.

- **Query Vector:** This is the word or token for which the attention weights are calculated. The Query vector specifies which sections of the input sequence should be prioritized. When you multiply word embeddings by the Query vector, you ask, “What should I pay attention to?”

- **Key Vector:** The set of words or tokens in the input sequence compared to the Query. The Key vector aids in identifying the important or relevant information in the input sequence. When you multiply word embeddings by the Key vector, you ask yourself, “What is important to consider?”

- **Value Vector:** It stores the information or features associated with each word or token in the input sequence. The Value vector contains the actual data that will be weighted and mixed in accordance with the attention weights calculated between the Query and Key. The Value vector answers the query, “What information do we have?”
Before the introduction of the transformer design, the attention mechanism was mainly used to compare two sections of a text. For example, the model could focus on different areas of the input article while generating the summary for a task like summarization.

The self-attention mechanism allowed the models to highlight the most significant parts of the text. It can be used in encoder-only or decoder-only models to construct a powerful input representation. The text can be translated into embeddings for encoder-only scenarios, but decoder-only models enable text generation.

The implementation of the multi-head attention mechanism substantially enhances its accuracy. In this setup, multiple attention components process the same information, with each head learning to focus on unique features of the text, such as verbs, nouns, numerals, and more, throughout the training and generation process.

## The Architecture In Action
Find the Notebook for this section at towardsai.net/book.
Seeing the architecture in action shows how the above components work in a pre-trained large language model, providing insight into their inner workings using the transformers Hugging Face library. You will learn how to load a pre-trained tokenizer to convert text into token IDs, followed by feeding the inputs to each layer of the network and investigating the output.

First, use AutoModelForCausalLM and AutoTokenizer to load the model and tokenizer, respectively. Then, tokenize a sample sentence that will be used as input in the following steps.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

inp = "The quick brown fox jumps over the lazy dog"
inp_tokenized = tokenizer(inp, return_tensors="pt")
print(inp_tokenized['input_ids'].size())
print(inp_tokenized)

torch.Size([1, 10])
{'input_ids': tensor([[    2,   133,  2119,  6219, 23602, 13855,    81,     
5, 22414,  2335]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

We load Facebook’s Open Pre-trained transformer model with 1.3B parameters (facebook/opt-1.3b) in 8-bit format, a memory-saving strategy for efficiently utilizing GPU resources. The tokenizer object loads the vocabulary required to interact with the model and is used to convert the sample input (inpvariable) to token IDs and attention mask. The attention mask is a vector designed to help ignore specific tokens. In the given example, all indices of the attention mask vector are set to 1, indicating that every token will be processed normally. However, by setting an index in the attention mask vector to 0, you can instruct the model to overlook specific tokens from the input. Also, notice how the textual input is transformed into token IDs using the model’s pre-trained dictionary.

Next, let’s examine the model’s architecture by using the .model method.

```python
print(OPT.model)

OPTModel(
  (decoder): OPTDecoder(
    (embed_tokens): Embedding(50272, 2048, padding_idx=1)
    (embed_positions): OPTLearnedPositionalEmbedding(2050, 2048)
    (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (layers): ModuleList(
      (0-23): 24 x OPTDecoderLayer(
        (self_attn): OPTAttention(
          (k_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
          (q_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
          (out_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
        )
        (activation_fn): ReLU()
        (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05,
elementwise_affine=True)
        (fc1): Linear8bitLt(in_features=2048, out_features=8192, bias=True)
        (fc2): Linear8bitLt(in_features=8192, out_features=2048, bias=True)
        (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
```

The decoder-only model is a common choice for transformer-based language models. As a result, we must use the decoder key to gain access to its inner workings. The layers key also reveals that the decoder component comprises 24 stacked layers with the same design. To begin, consider the embedding layer.

```python
embedded_input = OPT.model.decoder.embed_tokens(inp_tokenized['input_ids'])
print("Layer:\t", OPT.model.decoder.embed_tokens)
print("Size:\t", embedded_input.size())
print("Output:\t", embedded_input)

Layer:   Embedding(50272, 2048, padding_idx=1)
Size:      torch.Size([1, 10, 2048])
Output:  tensor([[[-0.0407,  0.0519,  0.0574,  ..., -0.0263, -0.0355, -0.0260],
         [-0.0371,  0.0220, -0.0096,  ...,  0.0265, -0.0166, -0.0030],
         [-0.0455, -0.0236, -0.0121,  ...,  0.0043, -0.0166,  0.0193],
         ...,
         [ 0.0007,  0.0267,  0.0257,  ...,  0.0622,  0.0421,  0.0279],
         [-0.0126,  0.0347, -0.0352,  ..., -0.0393, -0.0396, -0.0102],
         [-0.0115,  0.0319,  0.0274,  ..., -0.0472, -0.0059,  0.0341]]],
       device='cuda:0', dtype=torch.float16, grad_fn=<EmbeddingBackward0>)
```

The embedding layer is accessed via the decoder object’s .embed_tokens method, which delivers our tokenized inputs to the layer. As you can see, the embedding layer will convert a list of IDs of the size [1, 10] to [1, 10, 2048]. This representation will then be employed and transmitted through the decoder layers.

As mentioned before, the positional encoding component uses the attention masks to build a vector that conveys the positioning signal in the model. The positional embeddings are generated using the decoder’s .embed_positions method. As can be seen, this layer generates a unique vector for each position, which is then added to the embedding layer’s output. This layer adds positional information to the model.

```python
embed_pos_input = OPT.model.decoder.embed_positions(
    inp_tokenized['attention_mask']
)
print("Layer:\t", OPT.model.decoder.embed_positions)
print("Size:\t", embed_pos_input.size())
print("Output:\t", embed_pos_input)

Layer:   OPTLearnedPositionalEmbedding(2050, 2048)
Size:      torch.Size([1, 10, 2048])
Output:  tensor([[[-8.1406e-03, -2.6221e-01,  6.0768e-03,  ...,  1.7273e-02,
          -5.0621e-03, -1.6220e-02],
         [-8.0585e-05,  2.5000e-01, -1.6632e-02,  ..., -1.5419e-02,
          -1.7838e-02,  2.4948e-02],
         [-9.9411e-03, -1.4978e-01,  1.7557e-03,  ...,  3.7117e-03,
          -1.6434e-02, -9.9087e-04],
         ...,
         [ 3.6979e-04, -7.7454e-02,  1.2955e-02,  ...,  3.9330e-03,
          -1.1642e-02,  7.8506e-03],
         [-2.6779e-03, -2.2446e-02, -1.6754e-02,  ..., -1.3142e-03,
          -7.8583e-03,  2.0096e-02],
         [-8.6288e-03,  1.4233e-01, -1.9012e-02,  ..., -1.8463e-02,
          -9.8572e-03,  8.7662e-03]]], device='cuda:0', dtype=torch.float16, grad_fn=<EmbeddingBackward0>)
```

Lastly, the self-attention component! We can access the first layer’s self-attention component by indexing through the layers and using the .self_attn method. Also, examining the architecture’s diagram shows that the input for self-attention is created by adding the embedding vector to the positional encoding vector.

```python
embed_position_input = embedded_input + embed_pos_input
hidden_states, _, _ = OPT.model.decoder.layers[0].self_attn(embed_position_input)
print("Layer:\t", OPT.model.decoder.layers[0].self_attn)
print("Size:\t", hidden_states.size())
print("Output:\t", hidden_states)

Layer:   OPTAttention(
  (k_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
  (v_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
  (q_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
  (out_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=True)
)
Size:      torch.Size([1, 10, 2048])
Output:  tensor([[[-0.0119, -0.0110,  0.0056,  ...,  0.0094,  0.0013,  0.0093],
         [-0.0119, -0.0110,  0.0056,  ...,  0.0095,  0.0013,  0.0093],
         [-0.0119, -0.0110,  0.0056,  ...,  0.0095,  0.0013,  0.0093],
         ...,
         [-0.0119, -0.0110,  0.0056,  ...,  0.0095,  0.0013,  0.0093],
         [-0.0119, -0.0110,  0.0056,  ...,  0.0095,  0.0013,  0.0093],
         [-0.0119, -0.0110,  0.0056,  ...,  0.0095,  0.0013,  0.0093]]],
       device='cuda:0', dtype=torch.float16, grad_fn=<MatMul8bitLtBackward>)

```
The self-attention component includes the previously described query, key, and value layers and a final projection for the output. It accepts the sum of the embedded input and the positional encoding vector as input. In a real-world example, the model also supplies the component with an attention mask, allowing it to determine which parts of the input should be ignored or disregarded. (omitted from the sample code for clarity)

The remaining levels of the architecture employ nonlinearity (e.g., RELU), feedforward, and batch normalization.

💡 If you want to learn the transformer architecture in more detail and implement a GPT-like network from scratch, we recommend watching the video from Andrej Karpathy: Let’s build GPT: from scratch, in code, spelled out, accessible at towardsai.net/book.

## Transformer Model’s Design Choices
Find the Notebook for this section at towardsai.net/book.
The transformer architecture has proven its adaptability for a variety of applications. The original model was presented for the translation encoder-decoder task. Following the advent of encoder-only models such as BERT, the evolution of transformer design continued with the introduction of decoder-only networks in the first iteration of GPT models.

The variations are not limited to network architecture but also include differences in learning objectives. These different learning objectives significantly impact the model’s behavior and outcomes. Understanding these distinctions is critical for picking the best design for a given task and obtaining peak performance in various applications.

## The Encoder-Decoder Architecture
The full transformer architecture, often called the encoder-decoder model, consists of a number of encoder layers stacked together, linked to several decoder layers via a cross-attention mechanism. The architecture is exactly the same as we saw in the previous section.

These models are particularly effective for tasks that involve converting one sequence into another, like translating or summarizing text, where both the input and output are text-based. It’s also highly useful in multi-modal applications, such as image captioning, where the input is an image and the desired output is its corresponding caption. In these scenarios, cross-attention plays a crucial role, aiding the decoder in concentrating on the most relevant parts of the content throughout the generation process.

A prime illustration of this method is the BART pre-trained model, which features a bidirectional encoder tasked with forming a detailed representation of the input. Concurrently, an autoregressive decoder produces the output sequentially, one token after another. This model processes an input where some parts are randomly masked alongside an input shifted by one token. It strives to reconstitute the original input, setting this task as its learning goal. The provided code below loads the BART model to examine its architecture.

```python
from transformers import AutoModel, AutoTokenizer

BART = AutoModel.from_pretrained("facebook/bart-large")
print(BART)

BartModel(
  (shared): Embedding(50265, 1024, padding_idx=1)
  (encoder): BartEncoder(
    (embed_tokens): Embedding(50265, 1024, padding_idx=1)
    (embed_positions): BartLearnedPositionalEmbedding(1026, 1024)
    (layers): ModuleList(
      (0-11): 12 x BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05,
elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layernorm_embedding): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): BartDecoder(
    (embed_tokens): Embedding(50265, 1024, padding_idx=1)
    (embed_positions): BartLearnedPositionalEmbedding(1026, 1024)
    (layers): ModuleList(
      (0-11): 12 x BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05,
elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((1024,), eps=1e-05,
elementwise_affine=True)
        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layernorm_embedding): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
)
```

We are already familiar with most of the layers in the BART model. The model consists of encoder and decoder components, each with 12 layers. Furthermore, the decoder component, in particular, incorporates an additional encoder_attn layer known as cross-attention. The cross-attention component will condition the decoder output based on the encoder representations. We can use the transformers pipeline functionality and the fine-tuned version of this model for summarization.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sum = summarizer("""Gaga was best known in the 2010s for pop hits like “Poker Face” and avant-garde experimentation on albums like “Artpop,” and Bennett, a singer who mostly stuck to standards, was in his 80s when the pair met. And yet Bennett and Gaga became fast friends and close collaborators, which they remained until Bennett’s death at 96 on Friday. They recorded two albums together, 2014’s “Cheek to Cheek” and 2021’s “Love for Sale,” which both won Grammys for best traditional pop vocal album.""", min_length=20, max_length=50)


print(sum[0]['summary_text'])

Bennett and Gaga became fast friends and close collaborators.
They recorded two albums together, 2014's "Cheek to Cheek" and 2021's
"Love for Sale"
```

## The Encoder-Only Architecture
 
 Image

The overview of the encoder-only architecture with the attention and feed forward heads, taking the input, embedding it, going through multiple encoder blocks and its output is usually sent to either a decoder block of the transformer architecture or used directly for language understanding and classification tasks.

The encoder-only models are created by stacking many encoder components. Because the encoder output cannot be coupled to another decoder, it can only be used as a text-to-vector method to measure similarity. It can also be paired with a classification head (feedforward layer) on top to help with label prediction (also known as a Pooler layer in libraries like Hugging Face).

The absence of the Masked Self-Attention layer is the fundamental distinction in the encoder-only architecture. As a result, the encoder can process the full input at the same time. (Unlike decoders, future tokens must be masked out during training to avoid “cheating” when producing new tokens.) This characteristic makes them exceptionally well-suited for generating vector representations from a document, ensuring the retention of all the information.

The BERT article (or a higher quality variant like RoBERTa) introduced a well-known pre-trained model that greatly improved state-of-the-art scores on various NLP tasks. The model is pre-trained with two learning objectives in mind:

- Masked Language Modeling: obscuring random tokens in the input and trying to predict these masked tokens.
- Next Sentence Prediction: Present sentences in pairs and determine whether the second sentence logically follows the first sentence in a text sequence.

BERT = AutoModel.from_pretrained("bert-base-uncased")
print(BERT)

BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)

The BERT model employs the traditional transformer architecture with 12 stacked encoder blocks. However, the network’s output will be passed on to a pooler layer, a feed-forward linear layer followed by non-linearity that will construct the final representation. This representation will be used for other tasks like classification and similarity assessment. The code below uses a fine-tuned version of the BERT model for sentiment analysis:

classifier = pipeline("text-classification",
model="nlptown/bert-base-multilingual-uncased-sentiment")
lbl = classifier("""This restaurant is awesome.""")

print(lbl)

[{'label': '5 stars', 'score': 0.8550480604171753}]


The Decoder-Only Architecture

Image

The overview of the decoder-only architecture with the attention and feed forward heads. The input as well as recently predicted output goes into the model, is embedded, goes through multiple decoder blocks and produces the output probabilities for the next token.

Today’s Large Language Models mainly use decoder-only networks as their base, with occasional minor modifications. Due to the integration of masked self-attention, these models primarily focus on predicting the next token, which gave rise to the concept of prompting.

According to research, scaling up the decoder-only models can considerably improve the network’s language understanding and generalization capabilities. As a result, individuals can excel at various tasks just by employing varied prompts. Large pre-trained models, such as GPT-4 and LLaMA 2, may execute tasks like classification, summarization, translation, and so on by utilizing the relevant instructions.

The Large Language Models, such as those in the GPT family, are pre-trained with the Causal Language Modeling objective. It means the model attempts to predict the next word, whereas the attention mechanism can only attend to previous tokens on the left. This means the model can only anticipate the next token based on the previous context and cannot peek at future tokens, avoiding cheating.

gpt2 = AutoModel.from_pretrained("gpt2")
print(gpt2)

GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)

By looking at the architecture, you’ll discover the normal transformer decoder block without the cross-attention layer. The GPT family also uses distinct linear layers (Conv1D) to transpose the weights. (Please remember that this is not to be confused with PyTorch’s convolutional layer!) This design choice is unique to OpenAI; other large open-source language models employ the conventional linear layer. The provided code shows how the pipeline may incorporate the GPT-2 model for text production. It generates four possibilities to complete the statement, “This movie was a very.”

generator = pipeline(model="gpt2")
output = generator("This movie was a very", do_sample=True,
top_p=0.95, num_return_sequences=4, max_new_tokens=50, return_full_text=False)

for item in output:
  print(">", item['generated_text'])

>  hard thing to make, but this movie is still one of the most amazing 
shows I've seen in years. You know, it's sort of fun for a couple of
decades to watch, and all that stuff, but one thing's for sure —
>  special thing and that's what really really made this movie special," 
said Kiefer Sutherland, who co-wrote and directed the film's cinematography.
"A lot of times things in our lives get passed on from one generation to
another, whether
>  good, good effort and I have no doubt that if it has been released, 
I will be very pleased with it."

Read more at the Mirror.
>  enjoyable one for the many reasons that I would like to talk about here. 
First off, I'm not just talking about the original cast, I'm talking about
the cast members that we've seen before and it would be fair to say that
none of

 

💡 Please be aware that running the above code will yield different outputs due to the randomness involved in the generation process.

The Generative Pre-trained Transformer (GPT) Architecture
The OpenAI Generative Pre-trained Transformer (GPT) is a transformer-based language model. The ‘transformer’ component from its name relates to its transformer design, introduced in Vaswani et al.’s research paper “Attention is All You Need.”

Unlike traditional Recurrent Neural Networks (RNNs), which struggle with long-term dependencies due to the vanishing gradient problem, Long Short-Term Memory (LSTM) networks introduce a more complex architecture with memory cells that can maintain information over longer sequences. However, both RNNs and LSTMs still rely on sequential processing. In contrast, the transformer architecture abandons recurrence in favor of self-attention processes, significantly improving speed and scalability by enabling parallel processing of sequence data.

The GPT Architecture
The GPT series contains decoder-only models with a self-attention mechanism paired with a position-wise fully linked feed-forward network in each layer of the architecture.

Scaled dot-product attention is a self-attention technique that allows the model to assign a score of importance to each word in the input sequence while generating subsequent words. Additionally, “masking” within the self-attention process is a prominent element of this architecture. This masking narrows the model’s focus, prohibiting it from examining certain places or words in the sequence.

Image

Illustrating which tokens are attended to by masked self-attention at a particular timestamp. The whole sequence is passed to the model, but the model at timestep 5 tries to predict the next token by only looking at the previously generated tokens, masking the future tokens. This prevents the model from “cheating” by predicting and leveraging future tokens.

The following code implements the “masked self-attention” mechanism.

import numpy as np

def self_attention(query, key, value, mask=None):
    # Compute attention scores
    scores = np.dot(query, key.T)
    
    if mask is not None:
        # Apply mask by setting masked positions to a large negative value
        scores = scores + mask * -1e9
    
    # Apply softmax to obtain attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1,
keepdims=True)
    
    # Compute weighted sum of value vectors
    output = np.dot(attention_weights, value)
    
    return output

The initial step involves creating a Query, Key, and Value vector for every word in the input sequence. This is achieved through distinct linear transformations applied to the input vector. Essentially, it’s a simple feedforward linear layer that the model acquires through its training process.

Next, the model calculates attention scores by computing the dot product between the Query vector of each word and the Key vector of every other word. To ensure the model disregards certain phrases during attention, masking is applied by assigning significantly negative values to scores at specific positions. The SoftMax function then transforms these attention scores into probabilities, nullifying the impact of substantially negative values. Subsequently, each Value vector is multiplied by its corresponding weight and summed up to produce the output for the masked self-attention mechanism for each word.

Although this description illustrates the functionality of a single self-attention head, it’s important to note that each layer typically contains multiple heads, with numbers varying from 16 to 32, based on the specific model architecture. These multiple heads operate concurrently, significantly enhancing the model’s data analysis and interpretation capacity.

Causal Language Modeling
Large Language Models (LLMs) use self-supervised learning for pre-training on data with soft ground truth, eliminating the need for explicit labels for the model during training. This data can either be text that we already know the next words to predict or, for example, images with captions taken from Instagram. This permits LLMs to gain knowledge on their own. For example, utilizing supervised learning to train a summarizing model demands using articles and their summaries as training references. On the other hand, LLMs use the causal language modeling objective to learn from text data without requiring human-provided labels. Why is it called “causal”? Because the prediction at each step is purely based on previous steps in the sequence rather than future ones. 

💡 The procedure involves providing the model with a portion of text and instructing it to predict the next word.

After the model predicts a word, it is concatenated with the original input and presented to the model to predict the next token. This iterative process continues, with each newly generated token fed into the network. Throughout pre-training, the model progressively acquires an extensive understanding of language and grammar. Subsequently, the pre-trained model can be fine-tuned using a supervised method for various tasks or specific domains.

This approach offers an advantage over other methods by more closely replicating the natural way humans write and speak. Unlike masked language modeling, which introduces masked tokens into the input, causal language modeling sequentially constructs sentences one word at a time. This distinction ensures the model remains effective when processing real-world texts that do not include masked tokens.

Additionally, this technique allows the use of a wide range of high-quality, human-generated content from sources like books, Wikipedia, and news websites. Well-known datasets are readily accessible from platforms such as Hugging Face Hub.

MinGPT
There are various implementations of the GPT architecture, each tailored for specific purposes. While we will cover alternative libraries more suitable for production environments in upcoming chapters, it’s worth highlighting a lightweight version of OpenAI’s GPT-2 model, developed by Andrej Karpathy, called minGPT.

Karpathy describes minGPT as an educational tool designed to simplify the GPT structure. Remarkably, it is condensed into approximately 300 lines of code and utilizes the PyTorch library. Its simplicity makes it an excellent resource for gaining a deeper understanding of the internal workings of such models. The code is thoroughly described, providing clear explanations of the processes involved.

Three primary files are critical within the minGPT repository. The architecture is detailed in the model.py file. Tokenization is handled via the bpe.py file, which employs the Byte Pair Encoding (BPE) technique. The trainer.py file contains a generic training loop that may be used for any neural network, including GPT models. Furthermore, the demo.ipynb notebook shows the entire application of the code, including the inference process. This code is lightweight enough to run on a MacBook Air, allowing easy experimentation on a local PC. Those who prefer cloud-based solutions can fork the repository and utilize it in platforms such as Colab.

Introduction to Large Multimodal Models
Multimodal models are engineered to process and interpret diverse data types, or modalities, such as text, images, audio, and video. This integrated approach enables a more holistic analysis than models limited to a single data type, like text in conventional LLMs. For instance, augmenting text prompts with audio or visual inputs allows these models to comprehend a more intricate representation of information, considering factors like vocal nuances or visual contexts.

The recent surge of interest in LLMs has naturally extended to exploring LMMs’ potential, aiming to create versatile general-purpose assistants capable of handling a wide range of tasks.

Common Architectures and Training Objectives
By definition, multimodal models are intended to process numerous input modalities, such as text, images, and videos, and generate output in many modalities. However, a significant subset of currently popular LMMs primarily accept image inputs and can only generate text outputs.

These specialized LMMs frequently use pre-trained large-scale vision or language models as a foundation. They are known as ‘Image-to-Text Generative Models’ or visual language models (VLMs). They often conduct picture comprehension tasks such as question answering and image captioning. Examples include Microsoft’s GIT, SalesForce’s BLIP2, and DeepMind’s Flamingo.

Model Architecture
In the architecture of these models, an image encoder is utilized to extract visual features, followed by a standard language model that generates a text sequence. The image encoder might be based on Convolutional Neural Networks (CNNs), for instance, the ResNet, or it could use a transformer-based architecture, like the Vision Transformer (ViT).

There are two main approaches for training: building the model from scratch or utilizing pre-trained models. The latter is commonly preferred in advanced models. A notable example is the pre-trained image encoder from OpenAI’s CLIP model. In terms of language models, a wide range of pre-trained options are available, including Meta’s OPT, LLaMA 2, or Google’s FlanT5, which are instruction-trained.

Some models, like BLIP2, incorporate a novel element: a trainable, lightweight connection module that bridges the vision and language modalities. This approach, where only the connection module is trained, is cost-effective and time-efficient. Moreover, it demonstrates robust zero-shot performance in image understanding tasks.

Training Objective
LMMs are trained using an auto-regressive loss function applied to the output tokens. The concept of ‘picture tokens,’ similar to text tokenization, is introduced when employing a Vision Transformer architecture. This way, text can be separated into smaller units such as sentences, words, or sub-words for faster processing, and photographs can be segmented into smaller, non-overlapping patches known as ‘image tokens.’

In the Transformer architecture used by LMMs, specific attention mechanisms are key. Here, image tokens can ‘attend’ to one another, affecting how each is represented within the model. Furthermore, the creation of each text token is influenced by all the image and text tokens that have been generated previously.

Differences in Training Schemes
Despite having the same training objective, distinct language multimodal models (LMMs) have considerable differences in their training strategies. For training, most models, such as GIT and BLIP2, exclusively use image-text pairs. This method effectively establishes linkages between text and image representations but requires a large, curated dataset of image-text pairs.

On the other hand, Flamingo is designed to accept a multimodal prompt, which may include a combination of images, videos, and text, and generate text responses in an open-ended format. This capability allows it to perform tasks effectively, such as image captioning and visual question answering. The Flamingo model incorporates architectural advancements that enable training with unlabeled web data. It processes the text and images extracted from the HTML of 43 million web pages. Additionally, the model assesses the placement of images in relation to the text, using the relative positions of text and image elements within the Document Object Model (DOM).

The integration of different modalities is achieved through a series of steps. Initially, a Perceiver Resampler module processes spatiotemporal (space and time) features from visual data, like images or videos, which the pre-trained Vision Encoder processes. The Perceiver then produces a fixed number of visual tokens.

These visual tokens condition a frozen language model, a pre-trained language model that will not get updates during this process. The conditioning is made possible by adding newly initialized cross-attention layers incorporated with the language model’s existing layers. Unlike the other components, these layers are not static and updated during training. Although this architecture might be less efficient due to the increased number of parameters requiring training compared to BLIP2, it offers more sophisticated means for the language model to integrate and interpret visual information.

Few-shot In-Context-Learning
Flamingo’s flexible architecture allows it to be trained with multimodal prompts that interleave text with visual tokens. This enables the model to demonstrate emergent abilities, such as few-shot in-context learning, similar to GPT-3.

Open-sourcing Flamingo
As reported in its research paper, the advancements demonstrated in the Flamingo model mark a significant progression in Language-Multimodal Models (LMMs). Despite these achievements, DeepMind has yet to release the Flamingo model for public use.

Addressing this, the team at Hugging Face initiated the development of an open-source version of Flamingo named IDEFICS. This version is built exclusively with publicly available resources, incorporating elements like the LLaMA v1 and OpenCLIP models. IDEFICS is presented in two versions: the ‘base’ and the ‘instructed’ variants, each available in two sizes, 9 and 80 billion parameters. The performance of IDEFICS is comparable to the Flamingo model.

For training these models, the Hugging Face team utilized a combination of publicly accessible datasets, including Wikipedia, the Public Multimodal Dataset, and LAION. Additionally, they compiled a new dataset named OBELICS, a 115 billion token dataset featuring 141 million image-text documents sourced from the web, with 353 million images. This dataset mirrors the one described by DeepMind for the Flamingo model.

In addition to IDEFICS, another open-source replica of Flamingo, known as Open Flamingo, is publicly available. The 9 billion parameter model demonstrates a performance similar to Flamingo’s. The link to the IDEFICS playground is accessible at towardsai.net/book.

Instruction-tuned LMMs
As demonstrated by GPT-3’s emergent abilities with few-shot prompting, where the model could tackle tasks it hadn’t seen during training, there’s been a rising interest in instruction-fine-tuned LMMs. By allowing the models to be instruction-tuned, we can expect these models to perform a broader set of tasks and better align with human intents. This aligns with the work done by OpenAI with InstructGPT and, more recently, GPT-4. They have highlighted the capabilities of their latest iteration, the “GPT-4 with vision” model, which can process instructions using visual inputs. This advancement is detailed in their GPT-4 technical report and GPT-4V(ision) System Card.

Image

Example prompt demonstrating GPT-4’s visual input capability. The prompt requires image understanding. From the GPT-4 Technical Report.

Following the release of OpenAI’s multimodal GPT-4, there has been a significant increase in research and development of instruction-tuned Language-Multimodal Models (LMMs). Several research labs have contributed to this growing field with their models, such as LLaVA, MiniGPT-4, and InstructBlip. These models share architectural similarities with earlier LMMs but are explicitly trained on datasets designed for instruction-following.

Exploring LLaVA - An Instruction-tuned LMM
LLaVA, an instruction-tuned Language-Multimodal Model (LMM), features a network architecture similar to the previously discussed models. It integrates a pre-trained CLIP visual encoder with the Vicuna language model. A simple linear layer, which functions as a projection matrix, facilitates the connection between the visual and language components. This matrix, called W, is designed to transform image features into language embedding tokens. These tokens are matched in dimensionality with the word embedding space of the language model, ensuring seamless integration.

In designing LLaVA, the researchers opted for these new linear projection layers, lighter than the Q-Former connection module used in BLIP2 and Flamingo’s perceiver resampler and cross-attention layers. This choice reflects a focus on efficiency and simplicity in the model’s architecture.

This model is trained using a two-stage instruction-tuning procedure. Initially, the projection matrix is pre-trained on a subset of the CC3M dataset comprised of image-caption pairs. Following that, the model is fine-tuned end-to-end. During this phase, the projection matrix and the language model are trained on a specifically built multimodal instruction-following dataset for everyday user-oriented applications.

In addition, the authors use GPT-4 to create a synthetic dataset with multimodal instructions. This is accomplished by utilizing widely available image-pair data. GPT-4 is presented with symbolic representations of images during the dataset construction process, which comprises captions and the coordinates of bounding boxes. These COCO dataset representations are used as prompts for GPT-4 to produce training samples.

This technique generates three types of training samples: question-answer conversations, thorough descriptions, and complex reasoning problems and answers. The total number of training samples generated by this technique is 158,000.

The LLaVA model demonstrates the efficiency of visual instruction tuning using language-only GPT-4. They demonstrate its capabilities by triggering the model with the same query and image as in the GPT-4 report. The authors also describe a new SOTA by fine-tuning ScienceQA, a benchmark with 21k multimodal multiple-choice questions with substantial domain variety over three subjects, 26 themes, 127 categories, and 379 abilities.

Beyond Vision and Language
In recent months, image-to-text generative models have dominated the Large Multimodal Model (LMM) environment. However, other models include modalities other than vision and language. For instance, PandaGPT is designed to handle any input data type, thanks to its integration with the ImageBind encoder. There’s also SpeechGPT, a model that integrates text and speech data and generates speech alongside text. Additionally, NExT-GPT is a versatile model capable of receiving and producing outputs in any modality.

HuggingGPT is an innovative solution that works with the Hugging Face platform. Its central controller is a Large Language Model (LLM). This LLM determines which Hugging Face model is best suited for a task, selects that model, and then returns the model’s output.

Whether we are considering LLMs, LMMs, and all the types of models we just mentioned, one question remains: should we use proprietary models, open models, or open-source models?

To answer this question, we first need to understand each of these types of models.

Proprietary vs. Open Models vs. Open-Source Language Models
Language models can be categorized into three types: proprietary, open models, and open-source models. Proprietary models, such as OpenAI’s GPT-4 and Anthropic’s Claude 3 Opus, are only accessible through paid APIs or web interfaces. Open models, like Meta’s LLaMA 2 or Mistral’s Mixtral 8x7B, have their model architectures and weights openly available on the internet. Finally, open-source models like OLMo by AI2 provide complete pre-training data, training code, evaluation code, and model weights, enabling academics and researchers to re-create and analyze the model in depth.

Proprietary models typically outperform open alternatives because companies want to maintain their competitive advantage. They tend to be larger and undergo extensive fine-tuning processes. As of April 2024, proprietary models consistently lead the LLM rankings on the LYMSYS Chatbot Arena leaderboard. This arena continuously gathers human preference votes to rank LLMs using an Elo ranking system.

Some companies offering proprietary models, like OpenAI, allow fine-tuning for their LLMs, enabling users to optimize task performance for specific use cases and within defined usage policies. The policies explicitly state that users must respect safeguards and not engage in illegal activity. Open weights and open-source models allow for complete customization but require your own extensive implementation and computing resources to run. When checking for reliability, service downtime must be considered in proprietary models, which can disrupt user access.

When choosing between proprietary and open AI models, it is important to consider factors such as the needs of the user or organization, available resources, and cost. For developers, it is recommended to begin with reliable proprietary models during the initial development phase and only consider open-source alternatives later when the product has gained traction in the market. This is because the resources required to implement an open model are higher.

The following is a list of noteworthy proprietary and open models as of April 2024. The documentation links are accessible at towardsai.net/book.

Cohere LLMs
Cohere is a platform that enables developers and businesses to create applications powered by Language Models (LLMs). The LLM models offered by Cohere are classified into three primary categories - “Command,” “Rerank,” and “Embed.” The “Command” category is for chat and long context tasks, “Rerank” is for sorting text inputs by semantic relevance, and “Embed” is for creating text embeddings.

Cohere’s latest Command R model is similar to OpenAI’s LLMs and is trained using vast internet-sourced data. It is optimized for retrieval-augmented generation (RAG) systems and tool-use tasks. The Command R model has a context length of 128,000 tokens and is highly capable in ten major languages.

The development of these models is ongoing, with new updates and improvements being released regularly.

Users interested in exploring Cohere’s models can sign up for a Cohere account and receive a free trial API key. This trial key has no credit or time restriction; however, API calls are limited to 100 per minute, which is generally enough for experimental projects.

For secure storage of your API key, it is recommended to save it in a .env file, as shown below.

COHERE_API_KEY="<YOUR-COHERE-API-KEY>"

Then, install the cohere Python SDK with this command.

pip install cohere

You can now generate text with Cohere as follows.

import cohere
co = cohere.Client('<<apiKey>>')
response = co.chat(
  chat_history=[
    {"role": "USER", "message": "Who discovered gravity?"},
    {"role": "CHATBOT", "message": "The man who is widely credited with discovering gravity is Sir Isaac Newton"}
  ],
  message="What year was he born?", # perform web search before answering the question. You can also use your own custom connector.
  connectors=[{"id": "web-search"}]
)
print(response)

OpenAI’s GPT-3.5 and GPT-4
OpenAI currently offers two advanced Large Language Models, GPT-3.5 and GPT-4, each accompanied by their faster “Turbo” versions.

GPT-3.5, known for its cost-effectiveness and proficiency in generating human-like text, is competent for basic chat applications and other generative language tasks. The Turbo variant is faster and cheaper, making it an excellent choice for developers seeking cheap but performant LLMs. Although primarily optimized for English, it delivers commendable performance in various languages.

OpenAI provides its Language Model Models (LLMs) through paid APIs. The Azure Chat Solution Accelerator also uses the Azure Open AI Service to integrate these models in enterprise settings, focusing on GPT-3.5. This platform enhances moderation and safety, allowing organizations to establish a secure and private chat environment within their Azure Subscription. It provides a customized user experience, prioritizing privacy and control within the organization’s Azure tenancy.

OpenAI also offers GPT-4 and GPT-4 Turbo, representing the height of OpenAI’s achievements in LLMs and model multimodality. Unlike its predecessors, GPT-4 Turbo can process text and image inputs, although it only generates text outputs. The GPT-4 variant family is currently the state of the art regarding large model performance.

Like all current OpenAI models, GPT-4’s training specifics and parameters remain confidential. However, its multimodality represents a significant breakthrough in AI development, providing unequaled capabilities to understand and generate content across diverse formats.

Anthropic’s Claude 3 Models
Claude 3 is Anthropic’s latest family of Large Language Models (LLMs), setting new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models: Claude 3 Haiku, Claude 3 Sonnet, and Claude 3 Opus. Each successive model offers increasingly powerful performance, allowing users to select the best balance of performance, speed, and cost for their specific application.

As of April 2024, Claude 3 Opus is ranked among the best models on the LMSYS Chatbot Arena Leaderboard.

All Claude 3 models have a 200K token context window, capable of processing inputs up to 1 million tokens. The 1M token window will be available to select customers in the short term. The models demonstrate increased capabilities in analysis, forecasting, nuanced content creation, code generation, and conversing in non-English languages.

Claude 3 models incorporate techniques from Anthropic, such as Constitutional AI, where you use a language model with clear directives (a constitution) to guide your own model during training instead of relying on human feedback to reduce brand risk and aim to be helpful, honest, and harmless. Anthropic’s pre-release process includes significant “red teaming” to assess the models’ proximity to the AI Safety Level 3 (ASL-3) threshold. The Claude 3 models are easier to use than the previous generation, better at following complex instructions, and adept at adhering to brand voice and response guidelines.

Anthropic plans to release frequent updates to the Claude 3 model family and introduce new features to enhance their capabilities for enterprise use cases and large-scale deployments.

Google DeepMind’s Gemini
Google’s latest LLM, Gemini, is an advanced and versatile AI model developed by Google DeepMind. Gemini is a multimodal model that can process various formats, like text, images, audio, video, and code. This enables it to perform multiple tasks and understand complex inputs.

The model has three versions: Gemini Ultra for complex tasks and performance comparable to GPT-4; Gemini Pro, useful for a wide range of tasks; and Gemini Nano, a small LLM for on-device efficiency. You can get an API key to use and build applications with Gemini through the Google AI Studio or Google Vertex AI. They also recently announced Gemini Pro 1.5 with a context window of up to 1 million tokens, Gemini 1.5 Pro achieves the longest context window of any large-scale foundation model yet.

Meta’s LLaMA 2
LLaMA 2, a state-of-the-art LLM developed by Meta AI, was made publicly available on July 18, 2023, under an open license for research and commercial purposes.

Meta’s detailed 77-page publication outlines LLaMA 2’s architecture, facilitating its recreation and customization for specific applications. Trained on an expansive dataset of 2 trillion tokens, LLaMA 2 performs on par with GPT-3.5 according to human evaluation metrics, setting new standards in open-source benchmarks.

Available in three parameter sizes - 7B, 13B, and 70B - LLaMA 2 also includes instruction-tuned versions known as LLaMA-Chat.

Its fine-tuning employs both Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF), adopting an innovative method for segmenting data based on prompts for safety and helpfulness. Don’t worry if this sounds intimidating; we will discuss SFT and RLHF in depth in the next chapter.

The reward models are key to its performance. LLaMA 2 uses distinct safety and helpfulness reward models to assess response quality, achieving a balance between the two.

LLaMA 2 has made significant contributions to the field of Generative AI, surpassing other open innovation models like Falcon or Vicuna in terms of performance.

The LLaMA 2 models are available on the Hugging Face Hub. To test the meta-llama/Llama-2-7b-chat-hf model, you must first request access by filling out a form on their website.

Start by downloading the model. It takes some time as the model weighs about 14GB.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# download model
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

Then, we generate a completion with it. This step is time-consuming if you generate text using the CPU instead of GPUs!

# generate answer
prompt = "Translate English to French: Configuration files are easy to use!"
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=100)

# print answer
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

Falcon
The Falcon models, developed by the Technology Innovation Institute (TII) of Abu Dhabi, have captured significant interest since their release in May 2023. They are available under the Apache 2.0 License, which allows permission for commercial use.

The Falcon-40B model demonstrated notable performance, surpassing other LLMs like LLaMA 65B and MPT-7B. Falcon-7B, another smaller variant, was also released and is designed for fine-tuning on consumer hardware. It has half the number of layers and embedding dimensions compared to Falcon-40B, making it more accessible to a broader range of users.

The training dataset for Falcon models, known as the “Falcon RefinedWeb dataset,” is carefully curated and conducive to multimodal applications, maintaining links and alternative texts for images. This dataset, combined with other curated corpora, constitutes 75% of the pre-training data for the Falcon models. While primarily English-focused, versions like “RefinedWeb-Europe” extend coverage to include several European languages.

The instruct versions of Falcon-40B and Falcon-7B fine-tuned on a mix of chat and instruct datasets from sources like GPT4all and GPTeacher, show even better performance.

The Falcon models can be found on the Hugging Face Hub. In this example, we test the tiiuae/falcon-7b-instruct model. The same code used for the LLaMA model can be applied here by altering the model_id.

model_id = "tiiuae/falcon-7b-instruct"

Dolly
Dolly is an open-source Large Language Model (LLM) developed by Databricks. Initially launched as Dolly 1.0, it exhibited chat-like interactive capabilities. The team has since introduced Dolly 2.0, an enhanced version with improved instruction-following abilities.

A key feature of Dolly 2.0 is its foundation on a novel, high-quality instruction dataset named “databricks-dolly-15k.” This dataset comprises 15,000 prompt/response pairs tailored specifically for instruction tuning in Large Language Models. Uniquely, the Dolly 2.0 dataset is open-source, licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License, allowing for broad usage, modification, and extension, including commercial use.

Dolly 2.0 is built on the EleutherAI Pythia-12 b architecture, featuring 12 billion parameters. This enables it to display relatively high-quality instruction-following performance. Although smaller in scale compared to some models like LLaMA 70B, Dolly 2.0 achieves impressive results, thanks partly to its training on real-world, human-generated data rather than synthesized datasets.

Databricks’ models, including Dolly 2.0, are accessible on the Hugging Face Hub. The databricks/dolly-v2-3b model is available for testing. The same code used for the LLaMA model can be applied here by altering the model_id.

model_id = "databricks/dolly-v2-3b"

Open Assistant
The Open Assistant initiative focuses on democratizing access to high-quality Large Language Models through an open-source and collaborative model. This project distinguishes itself from other LLM open-source alternatives, which often come with restrictive licenses, by aiming to provide a versatile, chat-based language model comparable to ChatGPT and GPT-4 for commercial use.

At the core of Open Assistant is a commitment to openness and inclusivity. The project has compiled a significant dataset contributed by over 13,000 volunteers. This dataset includes more than 600,000 interactions, 150,000 messages, and 10,000 fully annotated conversation trees covering various topics in multiple languages. The project promotes community engagement and contributions, inviting users to participate in data collection and ranking tasks to further enhance the language model’s capabilities.

The Open Assistant models are available on Hugging Face, accessible via the Hugging Face demo or the official website.

While Open Assistant offers a broad range of functionalities, it does encounter some performance limitations, especially in fields like mathematics and coding, due to fewer training interactions in these areas. Generally, the model is proficient in producing human-like responses, though it is not unsusceptible to occasional inaccuracies.

Mistral LLMs
Mistral has released both Open and Proprietary models. In September 2023, Mistral released Mistral 7B, an open model with 7.3B parameters. It outperforms LLaMA 2 13B and LLaMA 1 34B models in various benchmarks and nearly matches CodeLLaMA 7B in code-related tasks.

Mixtral 8x7B, another open model released in December 2023, is a sparse mixture of expert models that outperforms LLaMA 2 70B with 6x faster inference. It has 46.7B parameters but uses only 12.9B per token, providing cost-effective performance. Mixtral 8x7B supports multiple languages, handles 32k token context, and excels in code generation. Mixtral 8x7B Instruct is an optimized version for instruction following.

In February 2024, Mistral AI introduced Mistral Large, their most advanced language proprietary model. It achieves strong results on commonly used benchmarks, making it among the best-ranked models generally available through an API, next to GPT-4 and Claude 3 Opus. Mistral Large is natively fluent in English, French, Spanish, German, and Italian and has a 32K token context window for precise information recall. It is available through La Plateforme and Azure.

Alongside Mistral Large, Mistral AI released Mistral Small, an optimized model for latency and cost that outperforms Mixtral 8x7B. Both Mistral Large and Mistral Small support JSON format mode and function calling, enabling developers to interact with the models more naturally and interface with their own tools.

Applications and Use-Cases of LLMs
Healthcare and Medical Research
Generative AI significantly enhances patient care, drug discovery, and operational efficiency within the healthcare sector.

In diagnostics, generative AI is making impactful strides with patient monitoring and resource optimization. The integration of Large Language Models into digital pathology has notably improved the accuracy of disease detection, including cancers. Additionally, these models contribute to automating administrative tasks, streamlining workflows, and enabling clinical staff to concentrate on crucial aspects of patient care.

The pharmaceutical industry has seen transformative changes due to generative AI in drug discovery. This technology has expedited the process, brought more precision to medicine therapies, reduced drug development times, and cut costs. This progress is opening doors to more personalized treatments and targeted therapies, holding great promise for patient care.

Medtech companies are also harnessing the potential of generative AI to develop personalized devices for patient-centered care. By incorporating generative AI into the design process, medical devices can be optimized for individual patient requirements, improving treatment outcomes and patient satisfaction.

For example, Med-PaLM, developed by Google, is an LLM designed to provide accurate answers to medical queries. It’s a multimodal generative model capable of processing various biomedical data, including clinical text, medical imagery, and genomics, using a unified set of model parameters. Another notable example is BioMedLM, a domain-specific LLM for biomedical text created by the Stanford Center for Research on Foundation Models (CRFM) and MosaicML.

Finance
LLMs like GPT are becoming increasingly influential in the finance sector, offering new ways for financial institutions to engage with clients and manage risks.

A primary application of these models in finance is the enhancement of customer interaction on digital platforms. Models are utilized to improve user experiences through chatbots or AI-based applications, delivering efficient and seamless customer support with real-time responses to inquiries and concerns.

LLMs are also making significant contributions to the analysis of financial time-series data. These models can offer critical insights for macroeconomic analysis and stock market predictions by leveraging extensive datasets from stock exchanges. Their ability to forecast market trends and identify potential investment opportunities is quite useful for making well-informed financial decisions.

An example of an LLM application in finance is Bloomberg’s development of BloombergGPT. This model, trained on a combination of general and domain-specific documents, demonstrates superior performance in financial natural language processing tasks without compromising general LLM performance on other tasks.

Copywriting
Language models and generative AI significantly impact the field of copywriting by offering robust tools for content creation.

The applications of generative AI in copywriting are diverse. It can accelerate the writing process, overcome writer’s block, and boost productivity, thereby reducing costs. Furthermore, it contributes to maintaining a consistent brand voice by learning and replicating a company’s language patterns and style, fostering uniformity in marketing efforts.

Key use cases include generating content for websites and blog posts, crafting social media updates, composing product descriptions, and optimizing content for search engine visibility. Additionally, generative AI plays a crucial role in creating tailored content for mobile applications, adapting it to various platforms and user experiences.

Jasper is an example of a tool that simplifies generating various marketing content utilizing LLMs. The users can choose from a set of predefined styles or capture the unique tone of a company.

Education
LLMs are increasingly valuable in online learning and personalized tutoring. By evaluating individual learning progress, these models provide tailored feedback, adaptive testing, and customized learning interventions.

To address teacher shortages, LLMs offer scalable solutions such as virtual teachers or the enhancement of para-teacher capabilities with advanced tools. This enables educators to transition into the roles of mentors and guides, offering individualized support and interactive learning experiences.

The capability of AI to analyze student performance data allows for the personalization of the learning experience, adapting to each student’s unique needs and pace.

An example of LLMs in the educational field is Khanmigo from Khan Academy. In this application, LLMs function as virtual tutors, providing detailed explanations and examples to enhance understanding of various subjects. Additionally, they support language learning by generating sentences for grammar and vocabulary practice, contributing significantly to language proficiency.

Programming
In programming, LLMs and generative AI are becoming indispensable tools, providing significant assistance to developers. Models such as GPT-4 and its predecessors excel at generating code snippets from natural language prompts, thereby increasing the efficiency of programmers. These models, trained on extensive collections of code samples, can grasp the context, progressively improving their ability to produce relevant and accurate code.

The applications of LLMs in coding are varied and valuable. They facilitate code completion by offering snippet suggestions as developers type, saving time and minimizing errors. LLMs are also used for generating unit tests and automating the creation of test cases, thereby enhancing code quality and benefiting software maintenance.

However, the use of generative AI in coding presents its challenges. While these tools can boost productivity, it is crucial for developers to thoroughly review the generated code to ensure it is free of errors or security vulnerabilities. Additionally, careful monitoring and validation are required for model inaccuracies.

A notable product leveraging LLMs for programming is GitHub Copilot. Trained on billions of lines of code, Copilot can convert natural language prompts into coding suggestions across various programming language.

Legal Industry
In the legal sector, LLMs and generative AI have proven to be useful resources, offering diverse applications tailored to the unique demands of the field. These models excel at navigating the intricacies of legal language, interpretation, and the ever-evolving landscape of law. They can significantly assist legal professionals in various tasks, such as offering legal advice, comprehending complex legal documents, and analyzing texts from court cases.

A crucial goal for all LLM applications in law is to minimize inaccuracies, commonly referred to as ‘hallucinations,’ which are a notable issue with these models. Incorporating domain-specific knowledge, either through reference modules or by drawing on reliable data from established knowledge bases, can enable these models to yield more accurate and trustworthy results.

Additionally, they can identify critical legal terms within user input and swiftly assess legal scenarios, enhancing their practical utility in legal contexts.

Risks and Ethical Considerations of Using LLMs in the Real World
Deploying Large Language Models (LLMs) for real-world applications introduces various risks and ethical considerations.

One notable risk is the occurrence of “hallucinations,” where models generate plausible yet false information. This can have profound implications, especially in sensitive fields such as healthcare, finance, and law, where accuracy is vital.

Another area of concern is “bias.” LLMs may unintentionally reflect and propagate the societal biases inherent in their training data. This could lead to unfair outcomes in critical areas like healthcare and finance. Tackling this issue requires a dedicated effort towards thorough data evaluation, promoting inclusivity, and continually working to enhance fairness.

Data privacy and security are also essential. LLMs have the potential to unintentionally memorize and disclose sensitive information, posing a risk of privacy breaches. Creators of these models must implement measures like data anonymization and stringent access controls to mitigate this risk.

Moreover, the impact of LLMs on employment cannot be overlooked. While they offer automation benefits, it’s essential to maintain a balance with human involvement to retain and value human expertise. Overreliance on LLMs without sufficient human judgment can be perilous. Adopting a responsible approach that harmonizes the advantages of AI with human oversight is imperative for effective and ethical use.

Recap
The transformer architecture has demonstrated its versatility in various applications. The original architecture was designed for sequence-to-sequence tasks (where a sequence is inputted and an output is generated based on it), such as translation. The next evolution of transformer architecture began with the introduction of encoder-only models like BERT, followed by the introduction of decoder-only networks in the first iteration of GPT models. However, several building blocks, like embedding layers and the attention mechanism, are shared on both the encoder and decoder components.

We introduced the model’s structure by loading a pre-trained model and extracting its important components. We also observed what happens behind the surface of an LLM, specifically, the model’s essential component: the attention mechanism. The self-attention mechanism is at the heart of the transformer model, calculating a weighted total of the embeddings of all words in a phrase.

Even though the transformer paper presented an efficient architecture, various architectures have been explored with minor modifications in the code, like altering the sizes of embeddings and the dimensions of hidden layers. Experiments have also demonstrated that moving the batch normalization layer before the attention mechanism improves the model’s capabilities. Remember that there may be minor differences in the design, particularly for proprietary models like GPT-3 that have yet to release their source code.

While LLMs may appear to be the final solution for any work, it’s important to remember that smaller, more focused models might deliver comparable outcomes while functioning more effectively. Using a simple model like DistilBERT on your local server to measure similarity may be more appropriate for specific applications while providing a cost-effective alternative to proprietary models and APIs.

The GPT family of models is an example of a decoder-only architecture. The GPT family has been essential to recent advances in Large Language Models, and understanding transformer architecture and recognizing the distinct characteristics of decoder-only models is critical. These models excel at tasks requiring language processing. In this debate, we analyzed their share components and the factors that characterize their architecture. Initially, GPT models were designed to complete input text sequentially, one token at a time. The intriguing question is how these autocompletion models evolved into powerful “super models” capable of following instructions and performing a wide range of tasks.

The recent surge of interest in LLMs has naturally extended to exploring LMMs’ potential, aiming to create versatile general-purpose assistants. In the architecture of these models, an image encoder is utilized to extract visual features, followed by a standard language model that generates a text sequence. Some of the most popular models that mix vision and language include OpenAI’s multimodal GPT-4, LLaVA, MiniGPT-4, and InstructBlip. Advanced LMMs can incorporate a broader range of modalities. These models generalize more on problems they’ve never seen before with instruction tuning.

Language models can be categorized into three types: proprietary, open models, and open-source models. Proprietary models, such as OpenAI’s GPT-4 and Anthropic’s Claude 3 Opus, are only accessible through paid APIs or web interfaces. Open models, like Meta’s LLaMA 2 or Mistral’s Mistral 7B, have their model architectures and weights openly available on the internet. Finally, open-source models like OLMo by AI2 provide complete pre-training data, training code, evaluation code, and model weights, enabling academics and researchers to re-create and analyze the model in depth. Some other examples include, the Falcon models by TII showing impressive performance and unique training data, Dolly 2.0 by Databricks featuring a high-quality instruction dataset and open licensing, and the Open Assistant initiative democratizing access to LLMs through community-driven development.

While LLMs have a transformative impact on various industries, issues such as hallucinations, biases, data privacy, and the impact AI on employment exist in real-world deployment.

 
