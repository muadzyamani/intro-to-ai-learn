# Introduction
## Why you need to know about artificial intelligence

-   Artificial intelligence (AI) systems are increasingly present in the workplace.
-   Applications include discovering pharmaceuticals, recommending products, and detecting fraud.
-   This course provides a high-level overview of AI concepts and technology.
-   It covers what makes a system "intelligent," widely used machine learning algorithms, and artificial neural networks.
-   The target audience includes business professionals, entrepreneurs, managers, and students who will interact with AI systems.
-   The role of a manager will evolve to include working with AI systems and data, similar to how they currently work with software developers.

---

# 1. What is Artificial Intelligence?
## Define general intelligence

-   **Human Intelligence is Diverse:** Humans exhibit various types of intelligence (e.g., linguistic, scientific, artistic), and there is no single standard.
-   **Computer Intelligence is Different:** Computers excel at specific tasks, particularly those involving set rules and pattern matching.
    -   They began beating humans at Checkers shortly after the first AI workshop in 1956.
    -   Computers have beaten the best human players in Chess for decades.
    -   Google's DeepMind defeated the top player in Go, a game with more possible moves than atoms in the universe.
-   **Lack of Understanding:** Despite their capabilities, these systems do not understand the purpose or meaning of the games they play. They are simply executing their specialized talent for following rules and matching patterns.
-   **Defining AI is Challenging:**
    -   A common definition: A system that shows behavior that could be interpreted as human intelligence.
    -   This definition is subjective. One person might find a chess program intelligent, while another sees a home assistant as intelligent.
    -   **Example:** In 2022, a Google engineer claimed a chatbot had a soul because it expressed a fear of being "switched off." Other engineers saw this as sophisticated language modeling and pattern matching designed to sound human.
-   **Key Takeaway for Organizations:**
    -   AI is most impressive and effective in domains with set rules and data.
    -   Organizations that work within well-defined spaces (like web search and e-commerce) are the first to benefit.
    -   To assess AI's potential impact, look for areas with significant pattern matching, set rules, and probabilities.

## The general problem-solver

-   **The General Problem Solver (GPS):** A program created in 1956 by Allen Newell and Herbert A. Simon.
-   **Physical Symbol System Hypothesis:** The core idea behind GPS.
    -   It proposed that intelligence could be achieved by programming a machine to connect symbols to meanings or actions (e.g., stop sign -> stop, letter A -> sound, sandwich -> eating).
-   **The Chinese Room Argument:** A thought experiment by philosopher John Searle (1980) to argue against the idea that symbol manipulation equals intelligence.
    -   **Scenario:** A person who doesn't speak Chinese is in a locked room. They receive Chinese symbols through a mail slot. Using a detailed rule book (a "phrase book"), they match the incoming symbols to the correct output symbols and send them back out.
    -   **Result:** A native Chinese speaker outside might believe they are having a real conversation with someone who understands Chinese.
    -   **Argument:** The person inside the room is simply matching patterns without any understanding of the language or the conversation. Searle argued that this is how computers operate—they mindlessly match symbols without true comprehension.
    -   Modern AI assistants like Siri or Cortana operate similarly, matching a user's question to a pre-programmed response.
-   **Limitations of Symbol Systems:**
    -   Symbolic systems were the foundation of AI for 25 years.
    -   The primary challenge was a **"combinatorial explosion."** The number of possible symbol combinations and rules grew too large to be programmed, making the "phrase book" impossibly big.

## Strong vs. weak AI

-   Philosopher John Searle categorized AI into two types: strong AI and weak AI.
-   **Strong AI:**
    -   A machine that exhibits the full range of human cognitive abilities and consciousness.
    -   It would have emotions, a sense of humor, creativity, and self-awareness.
    -   This is the type of AI typically seen in science fiction (e.g., C-3PO's fear, Commander Data's creativity).
    -   Strong AI is considered to be purely theoretical and does not exist today.
-   **Weak AI (or Narrow AI):**
    -   AI that is designed and trained for a specific, narrow task.
    -   Examples: Apple's Siri, language processing, sorting photos.
    -   This is where almost all current AI development and energy is focused.
-   **Expert Systems:**
    -   A common form of weak AI in the 1970s and 80s, based on symbolic systems. Also known as **GOFAI** (Good Old-Fashioned AI).
    -   Human experts create a detailed list of rules (if-then statements) to solve a complex problem.
    -   **Example (Medicine):** A program guides a nurse through a diagnosis by matching symptoms to a pre-programmed decision tree.
        -   If `cough`, then check `temperature`.
        -   If `cough` AND `temperature`, then check for `dehydration`.
        -   If `cough`, `temperature`, AND `dehydration`, then check for `bronchitis`.
    -   To an observer, the system might seem intelligent, but it's just following a script created by experts, much like the person in the Chinese Room.
    -   **Limitation:** Expert systems faced the same "combinatorial explosion" problem as other symbolic systems; it was impossible for experts to program every possible scenario.

## Chapter Quiz

**Question 1 of 3**

Why did some of the earliest artificial intelligence systems focus on board games such as checkers and chess?

-   **It's easiest to make a computer system seem intelligent when it's working with set rules and patterns.**
-   **Explanation:** Early computers had limited processing power. Board games provided a constrained environment with simple rules and patterns, which was ideal for these early AI systems.

**Question 2 of 3**

Luella seeks medical attention for chest pains. A nurse uses an artificial intelligence program to diagnose the cause. Why is this system likely not really intelligent?

-   **The program only matches her symptoms to steps in a system an expert created.**
-   **Explanation:** This is an example of a weak AI expert system. It doesn't "think" but rather follows a pre-programmed logic tree designed by human doctors. It lacks genuine understanding.

**Question 3 of 3**

You're a product manager who's in charge of building a weak AI expert system that will give tax advice. You're working with dozens of accountants who go through thousands of different taxpayer scenarios. When a customer asks a question, then the expert system will ask a follow-up question. It will do this until it makes a recommendation. What's one of the biggest challenges with this system?

-   **There will be too many tax combinations for the experts to cover with one system.**
-   **Explanation:** This illustrates the problem of "combinatorial explosion." In a complex domain like taxes, the number of possible scenarios, questions, and rules becomes so vast that it's practically impossible for experts to program every single one.

---

# 2. Popular Uses for Artificial Intelligence

## Predictive AI

-   For centuries, humans have sought to predict the future to gain power and wealth.
-   Businesses have long tried to predict customer behavior (e.g., showing ads for sports gear during a sporting event).
-   Predictive AI uses pattern matching and scales it up to millions of customers with high accuracy.
-   It works by analyzing vast amounts of historical data (like purchase history) to identify patterns and predict future actions.
-   **Common Examples:**
    -   Search engine results (Google, Bing)
    -   Product recommendations (Amazon)
    -   Song suggestions (Spotify, Apple Music)
-   Predictive AI has been the "quiet engine" driving the success of many of today's largest tech companies, including LinkedIn, which started by accurately predicting business contacts.
-   These systems excel because they can analyze huge datasets and identify subtle patterns that are difficult for humans to see.
-   The core principle is that prediction accuracy increases with the amount of available data.

## Generative AI

-   The concept is compared to alchemy: taking a common material (data) and transforming it into something valuable (new, generated content).
-   Generative AI takes massive amounts of diverse data and uses it to generate something entirely new.
-   **Predictive AI vs. Generative AI:**
    -   **Predictive AI:** Is narrow and task-specific. It uses a smaller amount of data to make a prediction for one purpose (e.g., a music recommender and a product recommender are two separate systems).
    -   **Generative AI:** Is broad and flexible. It's trained on huge, varied datasets and can perform many different tasks, including prediction, understanding, and content generation (e.g., text, images, video). It's like "One ring to rule them all."
-   **Example: Banking Chatbot**
    -   A **predictive** chatbot could answer specific, simple questions like, "What's the balance on this account?"
    -   A **generative** chatbot could handle more complex, general questions by making connections between different types of information, such as, "Will today's financial news affect my balance in my account?"
-   **Organizational Trade-off:**
    -   Choose a simpler **predictive system** for specific tasks that require less processing power.
    -   Choose a more complex **generative system** for flexibility and the ability to answer a wider range of questions, but be prepared for its high data and processing demands.
-   A generative AI system can typically do what a predictive system can, but the reverse is not true.

---

# 3. The Rise of Machine Learning

## Machine learning

-   The core idea is to create a computer that can learn on its own through observation, rather than being explicitly programmed for every possibility.
-   This approach was developed after earlier symbolic "expert systems" failed due to the "combinatorial explosion" of rules.
-   Instead of programming intelligence, computer scientists decided to program a system *to become intelligent* by learning from data.
-   **The Origin of Machine Learning:**
    -   In 1959, computer scientist Arthur Samuel created a checkers program that learned by playing against itself.
    -   It observed patterns and taught itself winning strategies without a human programming the moves.
    -   Samuel coined the term "machine learning" to describe this process.
-   **The Role of Data:**
    -   Early machine learning was limited by the lack of digital data. Data acts as the "senses" for these systems.
    -   The internet explosion in the 1990s provided the massive amounts of data ("fuel") needed for machine learning to advance.
-   **Key Advantage:** Machine learning systems can continuously learn, adapt, and improve as they are exposed to more data.
-   The main challenge for organizations today is figuring out how to use their vast amounts of data; machine learning provides the tools to find valuable insights within that data.

## Artificial neural networks

-   An artificial neural network (ANN) is a popular machine learning approach that mimics the structure of the human brain.
-   **Analogy: "Animal, Vegetable, or Mineral?" Game:** An ANN works by processing information through layers to narrow down possibilities and make a probabilistic guess (e.g., "64% chance this image is a cat").
-   **How ANNs Learn (Training):**
    1.  **Structure:** An ANN has an `input layer` (where data enters), one or more `hidden layers` (where processing occurs), and an `output layer` (where the guess is made).
    2.  **Input:** A labeled piece of data (e.g., a picture of a dog) is fed into the network.
    3.  **Guess:** The network processes the data through its layers and makes a prediction at the output layer.
    4.  **Compare & Adjust:** The network compares its guess to the correct label. It then works backward, adjusting the numerical "dials" (weights) in its neurons to reduce the error.
    5.  **Repeat:** This process is repeated with hundreds of thousands of examples until the network can consistently make accurate guesses.
-   **Important Distinction:** The network doesn't "understand" what a dog is in a human sense. It only recognizes the data as a specific, recognizable pattern of dots (pixels).
-   **Data Requirement:** Like all machine learning systems, ANNs need access to huge amounts of data to learn effectively.
-   **Key Benefit:** ANNs can train themselves to understand and recognize complex patterns within massive datasets.

## Chapter Quiz

**Question 1 of 1**

How does an artificial neural network learn?

-   **It looks at the data and makes guesses, then it compares those guesses to the correct answer.**
-   **Explanation:** An artificial neural network makes a probabilistic guess (e.g., it's 60% sure an image contains a dog) and then refines its internal parameters by comparing that guess to the correct answer, repeating this process thousands of times.

---

# 4. Common AI Systems

## Searching for patterns in data

-   Machine learning has become the dominant form of AI over the last 30 years because it excels at finding patterns in massive datasets.
-   The wide availability of digital data (images, video, audio) has "supercharged" machine learning systems.
-   Data acts as the "food" for machine learning; more data allows the machine to identify patterns more easily.
-   **Example:** A video training platform collects data on user interactions (e.g., fast-forwarding, watch duration). Machine learning algorithms can analyze this data from millions of users to find patterns in content engagement.
-   This data has immense business value for creating new products or improving existing ones.
-   Companies like Google and Microsoft built their businesses on using machines to interpret massive datasets.
-   **The "Black Box" Problem:** A major challenge is that humans often don't understand *how* a machine learning model identifies patterns. The process is not transparent.
-   This lack of transparency can be a significant problem in industries like insurance and healthcare, where systems might make decisions about health and safety that humans cannot comprehend or verify.
-   It's important to remember that artificial intelligence and human intelligence are not the same; they may reach similar conclusions but through very different processes.

## Robotics

-   Robotics is an area of AI focused on machines that perform physical tasks, such as manufacturing, food delivery, or self-driving vehicles.
-   **Traditional Robotics:** In the past, robots were highly specialized and programmed for specific, repetitive tasks (e.g., welding machines in an auto plant). They were not considered intelligent and could not adapt.
-   **AI-Powered Robotics:** Combining robotics with machine learning allows machines to adapt to their environment and learn new tasks.
-   **Example: Self-Driving Cars**
    -   A car cannot be explicitly programmed for every possible situation on the road.
    -   Instead, they use machine learning and artificial neural networks, which are fed constant data from sensors.
    -   The system learns patterns of successful driving by analyzing vast amounts of data. This is often called "training the network."
    -   Google considers this a "data problem" more than a robotics problem, as teaching the car *when* to turn is far more complex than teaching it *how* to turn.
-   **Simpler Robots:** Many robots today (like a Roomba) still use "Good Old Fashioned AI" (GOFAI)—symbolic, rule-based systems—because the cost of an error in the physical world is high. For critical tasks like distributing medication, a simpler, more predictable programmed system is often preferred.

## Natural language processing

-   Natural Language Processing (NLP) is the field of AI that allows humans to communicate with machines using their own natural language.
-   **Evolution of NLP:** It has moved beyond simple keyword matching (e.g., typing "recipe for Belgian waffles") to understanding nuance, context, and attributes in natural language (e.g., saying "Could you gimme a good recipe for those big, fluffy waffles?").
-   **How it Works:** Modern NLP uses machine learning and ANNs to analyze millions of conversations and identify patterns and relationships between words.
-   Companies like Google, Microsoft, and Apple offer free services (email, voicemail) partly to gather conversational data to train their NLP models.
-   NLP aims to understand context and meaning, not just words.
    -   **Example:** A search for "what is love?" has evolved. Early search engines gave a database-driven, literal response. Now, NLP helps provide a more thoughtful, human-centric response with poetry and romantic history.
-   Communication is central to human experience, so a machine would not be perceived as intelligent if it couldn't communicate naturally.

## The internet of things

-   The Internet of Things (IoT) refers to everyday objects with sensors that connect to the internet and communicate data (e.g., thermostats, smartwatches, doorbells).
-   **A Massive New Data Source:** IoT devices provide a constant stream of real-world data (location, medical stats, user habits).
-   **AI and IoT:** This combination is powerful because AI is needed to analyze the vast amount of data generated by IoT devices to find patterns and predict behavior.
-   **Examples:**
    -   An Alexa home assistant "hears" a conversation about a trip to Rome, and a Rome travel guide appears in the Amazon recommendations.
    -   Ring doorbells collect data on people walking by, which can be used for facial recognition and sold to police departments to create surveillance networks.
    -   A smartwatch EKG can detect health issues and upload data. Companies like Apple use machine learning to analyze millions of these EKGs to predict health problems.
-   IoT takes the pattern-finding strength of machine learning out of the digital world and applies it to the physical world, tracking offline behavior.

## Generative systems

-   While predictive AI is used for specific tasks, generative AI can be used when there is sufficient data and computing power for more general tasks.
-   The recent rise of generative AI is due to the availability of massive datasets and the falling cost of computing.
-   **Examples of Generative AI:**
    -   **DALL-E 2:** An OpenAI system released in 2022 that was trained on billions of images to generate new, photorealistic graphics from text prompts.
    -   **ChatGPT:** An OpenAI chatbot trained on a deep learning ANN. It is flexible and can answer general questions because it was trained on trillions of word connections from books, websites, social media, etc.
-   **How Generative AI Works (e.g., ChatGPT):**
    1.  It is trained on vastly more data than a traditional predictive AI system.
    2.  Humans then help refine the system to correct misinformation and filter out inappropriate content.
    3.  The system generates responses by predicting the next word in a sentence, one word at a time, based on patterns from billions of conversations.
-   Because of the high cost, generative AI is dominated by large companies. Most businesses will likely use these systems as a service rather than building their own from scratch.

## Chapter Quiz

**Question 1 of 3**

What type of impact does artificial intelligence have on robotics?

-   **AI systems can create robots that can more easily learn new tasks.**
-   **Correct:** Early robots were programmed for each task. AI, specifically machine learning, allows robots to learn from experience and adapt without explicit programming.

**Question 2 of 3**

The healthcare and medical insurance industries caution against using machine learning to search for patterns in data, and they do not want machines making decisions about a person's health. Why?

-   **They may be decisions that humans cannot understand.**
-   **Correct:** This refers to the "black box" problem. Since humans don't fully understand how the machine identifies patterns, it can lead to unexplainable and potentially erroneous decisions in critical areas like health.

**Question 3 of 3**

What impact will the Internet of Things (IoT) have on artificial intelligence?

-   **These devices will be a great new source of “real world” data.**
-   **Correct:** IoT devices connect the physical world to the digital world, providing AI systems with a wealth of new data about where people go and how they interact with their physical environment, enabling new types of predictions.

---

# 5. Learn from Data

## Labeled and unlabeled data

-   Machine learning can be understood through two primary learning strategies, analogous to learning a new skill like chess.
-   **Supervised Learning:**
    -   This is like having a tutor. A data scientist provides the machine with labeled data (the "correct answers").
    -   The system trains itself by trying to match the correct labels and adjusts when it makes a mistake.
    -   **Strength:** Can be very accurate for specific tasks.
    -   **Weakness:** Requires a knowledgeable expert to spend time creating high-quality labeled data.
    -   **Example:** Amazon could label 1,000 customers as "high spenders." A supervised learning system would then analyze the data of these customers to find common patterns that define a high spender.
-   **Unsupervised Learning:**
    -   This is like learning by quietly observing games in a public park. The system is given a large amount of unlabeled data and must find patterns and structures on its own.
    -   The system might not know the names or labels for things, but it can figure out relationships and strategies through observation.
    -   **Strength:** Does not require the manual effort of labeling data. Can uncover unexpected patterns.
    -   **Weakness:** Needs access to a massive amount of data to be effective.
    -   **Example:** An unsupervised system is given all of Amazon's customer data. It might discover a surprising correlation on its own, such as customers who buy chessboards are also likely to buy expensive kitchen appliances.

## Massive datasets

-   Traditional programming, which relies on explicit instructions, fails with AI because there are too many possible combinations to program.
-   Machine learning switches the model: instead of programmers inputting instructions, they input data and let the machine learn the patterns.
-   **Supervised Learning Process:**
    1.  **Split the Data:** The available data is split into two parts:
        -   **Training Set:** A smaller, labeled portion of data that the machine uses to learn.
        -   **Test Set:** A much larger, often unlabeled, dataset used to see how well the trained machine performs on new data.
    2.  **Train the Algorithm:** Machine learning algorithms use statistics to find relationships and patterns within the training data.
    3.  **Test the Model:** Once the algorithm is accurate on the training set, it is applied to the larger test set to validate its performance.
-   **Example: Spam Detection**
    -   **Task:** Classify emails as either "Spam" or "Regular Message." This is a **Binary Classification** problem.
    -   **Training Set:** 10,000 emails, with 1,000 manually labeled as "Spam."
    -   **Algorithm:** The machine learns statistical patterns (e.g., emails with words like "Lucky," "Winner," or "Congratulations" are more likely to be Spam).
    -   **Test Set:** 1 million unlabeled emails. The trained model is used to identify Spam within this massive set.

## Data models

-   A **Data Model** is an abstraction of what the AI system has learned from all its training and data. It's the core concepts the machine has extracted.
-   Data models are never "finished." Machine learning teams constantly work to improve the model's accuracy (e.g., from 99% to 99.9%).
-   **Human Analogy:** Humans use data models to understand the world. Your "driving model" lets you operate an unfamiliar car, and your "bird model" helps you recognize a bird you've never seen before. The more data you process (experience), the more accurate your model.
-   **Predictive AI and Data Models:**
    -   Data models are central to predictive AI.
    -   Each specific predictive task requires its own specialized data model.
    -   Training to be a better driver (improving your "driving model") will not make you better at identifying birds (your "bird model").
-   **Generative AI vs. Data Models:**
    -   Generative AI is more flexible and needs much more data, combining knowledge from many different domains.
    -   It doesn't use a single, narrow data model in the same way as predictive AI.
    -   **Example:** To generate a story about driving a sports car with Big Bird, a GenAI system needs to access knowledge about cars, birds, and television shows simultaneously.
-   Data models remain the core of AI in most organizations because predictive AI offers powerful, accurate results for specific, well-defined problems.

## Chapter Quiz

**Question 1 of 2**

A new online camping goods store wants to find connections between products customers buy and other products they might buy. Why would the company use unsupervised learning?

-   **It does not yet have enough customers to make supervised learning meaningful.**
-   **Correct:** Supervised learning requires a large, established set of labeled data (e.g., "customers who bought X also bought Y"). A new store lacks this historical data, making unsupervised learning a better choice to discover initial patterns.

**Question 2 of 2**

You're a preschool worker and you want to teach your class the letters in the alphabet. So you draw the letter “B” on the board. Then you ask the two-year-old students to find a block with that same letter. Some of the students correctly find the blocks with the letter “B”, but some of the students confuse the letter “B” with the letter “D.” So the incorrect students compare their block to the letter “B” on the board, recognize the error and then decide to get another block. What type of learning is this?

-   **supervised learning**
-   **Correct:** This is an example of supervised learning because there is labeled data (the letter "B" on the board) that serves as the "correct answer." The students (the "system") make a guess by picking a block, compare it to the label, and adjust their strategy if they are wrong.

---

# 6. Identify Patterns

## Classify data

-   Humans naturally classify things to stay organized (e.g., putting documents into folders, separating contacts). Businesses do the same (e.g., classifying frequent flyers or high-spending customers).
-   **Binary Classification:** This is one of the most popular and powerful uses of supervised machine learning. It involves classifying data into one of two possible outcomes (e.g., Yes/No, Fraud/Not Fraud, Spam/Not Spam).
-   **Supervised Learning is Key:** Classification relies on supervised learning, which means it requires a large amount of labeled data for the training set.
-   **Example: Credit Card Fraud Detection**
    1.  Credit card companies start with a training set of tens of thousands of transactions that have been manually labeled as fraudulent.
    2.  A machine learning algorithm is trained on this data to recognize the patterns of fraud.
    3.  Once trained, the system can classify new, incoming transactions as either "fraud" or "not fraud."
-   **The Main Challenge:** The biggest difficulty is acquiring a massive amount of high-quality, pre-classified (labeled) data for the training set.
    -   This process is labor-intensive and may need to be repeated if the system isn't accurate enough.
-   Even with years of development, these systems are not perfect and are constantly being re-trained to improve their accuracy.
-   To a machine learning system, different problems like fraud detection, spam filtering, and purchase prediction are all the same task: classifying labeled data into predefined categories.

## Cluster data

-   Clustering is used when you don't have access to massive amounts of labeled data or when you want the system to discover its own groupings.
-   **Clustering:** An unsupervised learning technique where the machine creates its own groups (clusters) of data based on inherent patterns.
-   **Classification vs. Clustering:**
    -   **Classification (Supervised):** Sorts data into predefined, *human-created* categories.
    -   **Clustering (Unsupervised):** Groups data into *machine-created* clusters based on what it observes.
-   **Example: Online Shopping**
    -   A "Frequently bought together" feature is an example of clustering. An unsupervised system analyzes purchasing histories and creates clusters of items that are often bought in the same transaction.
-   **Analogy: Halloween Candy**
    -   **Classification (Supervised):** A child sorts their candy into known categories like "chocolate," "peanut butter," and "gummies" with a parent's help (the "supervisor").
    -   **Clustering (Unsupervised):** The child receives a bag of foreign candy (unlabeled data). They must study the candy and create their own clusters based on attributes like size, color, or even something unexpected like a "perfume candy" cluster.
-   **Advantage of Clustering:** It can be used with the vast amounts of unlabeled data available in the world and can reveal surprising, valuable patterns that a human would never have considered. Companies like Amazon and Netflix use clustering to group friends, search histories, and buying habits.

## Reinforcement learning

-   Used when the goal is to go beyond simple clustering and encourage discovery or optimize a strategy over time.
-   **Reinforcement Learning:** A type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties. The goal is to maximize the cumulative reward.
-   **Example: Music Recommendation Systems (Spotify Discover Weekly)**
    -   Unsupervised learning might just cluster songs that are "frequently bought together."
    -   Reinforcement learning aims to help you *discover* new music.
    -   The algorithm gets a "tiny digital reward" when you click on a recommended song. The longer you listen, the more the reward increases.
    -   The system learns to find patterns that maximize its "reward," leading to better, more creative recommendations.
-   **Q-Learning:**
    -   A popular reinforcement learning algorithm. The "Q" represents the "quality" of an action in a particular state.
    -   The system's goal is to learn a strategy that maximizes the Q-value over time.
    -   You can think of the Q-value as the system's "bank account," which it tries to grow by making better and better decisions.
-   Reinforcement learning is ideal when an organization wants a system that can think creatively and develop strategies, rather than just group existing data.

## Chapter Quiz

**Question 1 of 2**

What is one of the greatest challenges with supervised learning binary classification?

-   **You need a lot of pre-classified or labeled data for the training set.**
-   **Correct:** Supervised learning is entirely dependent on having a large, high-quality dataset where the correct answers have already been labeled by humans. Acquiring this data can be difficult and expensive.

**Question 2 of 2**

Why might you want to use reinforcement learning instead of unsupervised learning?

-   **Reinforcement learning allows the machine to make predictions and create strategies instead of just clustering the data.**
-   **Correct:** Unsupervised learning is good for finding existing groups in data. Reinforcement learning is more advanced; it develops strategies to achieve a goal (like maximizing a reward), allowing it to "invent" new ways to encourage customer behavior or solve a problem.

---

# 7. Machine Learning Algorithms

## Common algorithms

-   Machine learning is not one single program but a collection of different algorithms, many borrowed from statistics.
-   These algorithms are like a chef's kitchen tools; each has a primary use, but they can be combined creatively to solve problems.
-   **Example: A company using both supervised and unsupervised learning.**
    1.  **Supervised Learning (Classification):** First, they used binary classification to divide customers into two groups: those who use promotions and those who don't.
    2.  **Unsupervised Learning (Clustering):** Next, they took the "promotion users" group and applied unsupervised learning to find new patterns. The machine created its own clusters and identified a small, valuable subgroup of "promotion super users."
    3.  **Result:** The company then targeted this super-user group with more promotions, increasing its success rate.
-   Data science teams mix and match algorithms based on the data and the problem. Each algorithm has different strengths, weaknesses, and processing power requirements.

## K-nearest neighbor

-   **K-Nearest Neighbor (KNN)** is a common supervised machine learning algorithm used for **multi-class classification** (where there are more than two possible outcomes).
-   **Core Idea:** It classifies new, unknown data by looking at the known data points ("neighbors") that are closest to it. The goal is to minimize the distance between the unknown point and its neighbors.
-   **Analogy: Classifying a dog breed at an animal shelter.**
    -   An unknown dog is compared to dogs of known breeds.
    -   The unknown dog is classified based on the breed of the dogs it most closely resembles in features like face shape and hair color.
-   **How it Works:**
    1.  **Plot Data:** Known, labeled data is plotted on a graph using key features as predictors (e.g., X-axis for weight, Y-axis for hair length).
    2.  **Add Unknown Point:** The new, unlabeled data point is added to the graph.
    3.  **Find Neighbors:** The algorithm identifies the 'k' nearest labeled data points to the new point. 'K' is a number you choose (e.g., 5). The distance is often calculated using **Euclidean Distance**.
    4.  **Classify by Majority:** The new point is assigned to the class that is most common among its 'k' nearest neighbors. (e.g., if 3 of the 5 nearest neighbors are Shepherds, the unknown dog is classified as a Shepherd).
-   **Use Case:** Commonly used in finance to identify promising stocks by comparing them to the historical performance of similar stocks.

## K-means clustering

-   **K-Means Clustering** is a common unsupervised machine learning algorithm used to group data into a specified number ('k') of clusters.
-   **Core Idea:** The algorithm iteratively organizes data points into 'k' clusters by finding the best central point (**centroid**) for each cluster.
-   **Analogy: Clustering dogs at a shelter into 3 groups for 3 new shelters.**
    -   The 'k' in k-means would be 3.
-   **How it Works:**
    1.  **Initialize:** The algorithm starts by randomly selecting 'k' data points to be the initial centroids.
    2.  **Assign:** Each data point is assigned to the cluster of its nearest centroid.
    3.  **Update:** The centroid of each cluster is recalculated to be the mean (average) position of all points in that cluster.
    4.  **Iterate:** Steps 2 and 3 are repeated until the centroids stop moving significantly, and the clusters become stable.
-   **Challenges:**
    -   It forces the data into exactly 'k' clusters, even if the data doesn't naturally group that way.
    -   It can be sensitive to **outliers** (an isolated data point will still be forced into a cluster).
-   **Use Case:** Retailers use it to cluster customers into groups like "loyal customers," "regular customers," and "lowest-price shoppers" to create targeted marketing strategies.

## Regression

-   **Regression Analysis** is a supervised machine learning algorithm used to understand the relationship between variables and predict continuous outcomes by identifying trends.
-   **Core Idea:** It looks at the relationship between **predictors** (input variables) and an **outcome**. It tells you *that* a trend exists, but not *why*.
-   **Analogy: Seasonal car sales.**
    -   A company observes that people buy more convertibles in the summer and more trucks in the winter. Regression helps quantify this trend to predict future sales.
-   **How it Works:**
    -   Data is plotted on an X-Y diagram (e.g., X-axis for months, Y-axis for sales).
    -   A **trend line** is drawn through the data to represent the relationship between the variables.
    -   The more data you have, the more accurate the trend line and the better the predictions.
-   **Use Case:** Large retailers like Walmart use regression to predict seasonal demand for products (e.g., more Pop-Tarts in summer) to optimize stock levels.
-   **Note:** Some debate whether regression is "true" machine learning, as it's more about predicting based on statistical trends than learning entirely new concepts.

## Naive Bayes

-   **Naive Bayes** is a popular and powerful supervised machine learning algorithm for classification.
-   **Core Idea ("Naive"):** It's called "naive" because it makes the simplifying assumption that all predictors (features) are **independent** of one another.
-   **Analogy: Classifying a dog's breed using multiple features.**
    -   Even though a dog's height and weight are related, Naive Bayes treats them as two completely separate, independent pieces of evidence.
-   **How it Works (Class Predictor Probability):**
    1.  For an unknown item, the algorithm looks at **each predictor individually**.
    2.  It calculates the probability that the item belongs to each possible class, based on that single predictor.
    3.  This is done for all predictors.
    4.  The probabilities are combined to make a final classification, assigning the item to the class with the highest overall probability.
-   **Use Case:** Banks use it for fraud detection by looking at many independent transaction predictors (time, location, amount, etc.). Cybersecurity firms use it to detect threats by analyzing independent network traffic features.
-   **Key Strength:** Because it makes few assumptions, it can effectively handle a huge number of predictors, which often makes it more accurate.

## Chapter Quiz

**Question 1 of 2**

You work for a company that’s selling electric cars to consumers. The company wants to get the maximum amount of value from its advertising dollars. So it wants to ramp up advertising when it thinks that customers would be most interested in purchasing an electric car. Your data science team wants to create a regression analysis based on fuel prices. How might this look on an XY diagram?

-   **Create a trendline with fuel prices along the X axis and electric car sales on the Y axis.**
-   **Correct:** Regression analysis involves plotting two variables on an XY diagram to see if a trendline emerges. A clear trendline indicates a predictable relationship, such as electric car sales rising as fuel prices rise.

**Question 2 of 2**

How is K Nearest Neighbor like the old saying, “birds of a feather flock together?”

-   **Classify unknown data against the closest data that you do know.**
-   **Correct:** The saying implies that things that are similar tend to group together. KNN works on this principle by classifying an unknown item based on the classification of the known items that are most similar (closest) to it.

---

# 8. Fit the Algorithm

## Select the best algorithm

-   Just as a chef uses a variety of tools, a data science team uses a variety of machine learning algorithms.
-   **Ensemble Modeling:** The practice of combining multiple machine learning algorithms or models to produce a better outcome than any single model could on its own. It's like a musical ensemble where a group performs together.
-   **Popular Ensemble Techniques:**
    -   **Bagging:** Using several versions of the *same* machine learning algorithm on different random subsets of the data. The results are then aggregated (e.g., averaged) to create a more accurate final prediction.
    -   **Boosting:** Building models sequentially, where each new model tries to correct the errors of the previous one. The results are boosted step-by-step.
    -   **Stacking:** Using several *different* machine learning algorithms and "stacking" them. The predictions from the first layer of models are used as input for a final model that makes the ultimate prediction.
-   **Example: Retail Checkout Items**
    -   A home improvement retailer wants to decide what to put near the checkout.
    -   They could use **bagging** with KNN on data from different stores to find an average, stable trend.
    -   They could also use **stacking**. First, use KNN to find what's bought with a hammer (e.g., nails). Then, use Naive Bayes on top of that to find less obvious, uncorrelated items that hammer-buyers also purchase (e.g., chocolate bars).
-   The creativity of the data science team in mixing and matching these algorithms determines the quality of the insights.

## Follow the data

-   One of the biggest challenges in machine learning is finding the right balance between bias and variance.
-   **Bias:** The gap between the model's predicted value and the actual, true value. High bias means the model is consistently wrong in the same direction.
    -   **Darts Analogy:** All the darts land close together, but in the wrong spot (e.g., upper right corner instead of the bullseye).
-   **Variance:** How scattered or spread out the model's predictions are for a given data point. High variance means the predictions are all over the place.
    -   **Darts Analogy:** The darts are spread all over the dartboard, not clustered together.
-   **The Goal:** The ideal model has **low bias and low variance** (all darts are clustered tightly in the bullseye).
-   **Bias-Variance Trade-off:** This is a fundamental concept in machine learning. Generally, actions taken to decrease bias will increase variance, and actions taken to decrease variance will increase bias.
-   The machine learning algorithm must "follow the data" and turn the knobs of bias and variance to find the optimal trade-off that produces the best possible predictions.

## Overfitting and underfitting

-   **Underfitting:** The model is too simple to capture the underlying patterns in the data. It performs poorly on both the training data and the new test data.
    -   **Analogy:** A simple rule for a child ("always brush your teeth and shower before bed") works at home but fails in other situations (like naps at preschool).
    -   **Cause:** Often happens when the model has high bias.
-   **Overfitting:** The model is too complex and learns the training data too well, including its noise and random fluctuations. It performs perfectly on the training data but fails to generalize to new, unseen test data.
    -   **Analogy:** Adding too many complex exceptions to the child's simple rule, making it confusing and hard to follow.
    -   **Cause:** Often happens when the model has high variance.
-   **Example: Home Value Prediction**
    -   **Underfitting:** A model uses only four simple predictors (square footage, location, bedrooms, bathrooms). The rule is too simple and won't be accurate for the wide variety of homes on the market.
    -   **Overfitting:** To fix this, data scientists add many more complex predictors (quality of view, modern appliances, wood floors). The model becomes too complex and might start fitting to noise in the data, making it less accurate on new homes.
-   The key is to find a compromise—a model that is complex enough to capture the true patterns but simple enough to generalize to new data, balancing the risk of underfitting and overfitting.

## Chapter Quiz

**Question 1 of 3**

What is ensemble modeling?

-   **This is when you use a mix of different machine learning algorithms or data to improve the outcome.**
-   **Correct:** Ensemble modeling combines several machine learning models (either the same type with different data, or different types) to produce more accurate and robust predictions than a single model.

**Question 2 of 3**

How does the bias-variance trade-off affect machine learning?

-   **If the machine makes a change to one, it must consider how the other is affected.**
-   **Correct:** The relationship is a trade-off. Improving one (e.g., lowering bias) often makes the other worse (e.g., increases variance), so they must be balanced carefully.

**Question 3 of 3**

You work for a credit card company that's trying to do a better job identifying fraudulent transactions. So your team uses unsupervised learning to create clusters of transactions that are likely to be fraudulent. The machine identified that when customers are buying electronics it's much more likely to be a fraudulent transaction. So you use this model for your new fraud detection system. Then customers started to complain that they couldn't use their credit cards to purchase any electronics. What is the challenge with your model?

-   **You underfit the model to the data, the simple rule made too many inaccurate predictions.**
-   **Correct:** The model learned a rule that was too simple ("all electronics purchases are likely fraud"). This rule worked on the training data but failed to generalize to the real world, where most electronics purchases are legitimate. This is a classic case of underfitting.

---
