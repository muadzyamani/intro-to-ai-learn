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
