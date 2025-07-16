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