# Prompt Engineering Complete Guide

Master the art of crafting effective prompts for LLMs - techniques, patterns, and best practices.

---

## What is Prompt Engineering?

Prompt engineering is the practice of designing inputs to LLMs to get desired outputs. It's about **communicating clearly** with AI models.

```
Bad Prompt → Vague/Wrong Output
Good Prompt → Precise/Useful Output
```

---

## Core Principles

### 1. Be Specific
```
❌ "Write about dogs"
✓ "Write a 200-word paragraph about Golden Retrievers as family pets,
   focusing on their temperament and exercise needs"
```

### 2. Provide Context
```
❌ "Fix this code"
✓ "Fix this Python function that should return the sum of even numbers,
   but currently returns 0 for all inputs:
   [code here]"
```

### 3. Define the Format
```
❌ "List some programming languages"
✓ "List 5 programming languages in this format:
   1. [Language] - [Primary use case]"
```

### 4. Set Constraints
```
❌ "Explain quantum computing"
✓ "Explain quantum computing in 3 sentences,
   using no technical jargon, suitable for a 10-year-old"
```

---

## Prompting Techniques

### 1. Zero-Shot Prompting

Ask directly without examples.

```
Prompt: "Classify the sentiment of this review as positive, negative, or neutral:
'The food was okay but the service was terrible.'"

Output: "Negative"
```

**When to use:** Simple tasks the model already knows how to do.

---

### 2. One-Shot Prompting

Provide one example.

```
Prompt: "Convert the sentence to past tense:

Example:
Input: I eat an apple
Output: I ate an apple

Input: She runs to school
Output:"

Output: "She ran to school"
```

**When to use:** When format demonstration helps.

---

### 3. Few-Shot Prompting

Provide multiple examples to establish a pattern.

```
Prompt: "Classify the intent of customer messages:

Message: 'I want to cancel my subscription'
Intent: cancellation

Message: 'How do I reset my password?'
Intent: account_help

Message: 'Your product is amazing!'
Intent: feedback_positive

Message: 'I haven't received my order yet'
Intent:"

Output: "order_status"
```

**When to use:** Complex classification, specific formats, consistent style.

**Tips:**
- Use 3-5 diverse examples
- Cover edge cases in examples
- Keep examples consistent in format

---

### 4. Chain-of-Thought (CoT) Prompting

Ask the model to reason step-by-step.

```
Prompt: "Solve this problem step by step:

A store sells apples for $2 each. If you have $15 and want to buy
as many apples as possible while keeping at least $3 for bus fare,
how many apples can you buy?"

Output:
"Let me solve this step by step:
1. Total money: $15
2. Money needed for bus: $3
3. Money available for apples: $15 - $3 = $12
4. Price per apple: $2
5. Number of apples: $12 ÷ $2 = 6

Answer: 6 apples"
```

**Trigger phrases:**
- "Let's think step by step"
- "Solve this step by step"
- "Walk me through your reasoning"
- "Explain your thought process"

---

### 5. Zero-Shot CoT

Add "Let's think step by step" without examples.

```
Prompt: "If John has 3 sisters and each sister has 2 brothers,
how many brothers does John have? Let's think step by step."

Output:
"Let's think step by step:
1. John has 3 sisters
2. Each sister has 2 brothers
3. John is one of those brothers
4. So there must be 2 brothers total (including John)
5. Therefore, John has 1 brother (the other one besides himself)

Answer: 1 brother"
```

---

### 6. Self-Consistency

Generate multiple CoT paths, take majority vote.

```python
# Generate 5 responses with temperature > 0
responses = []
for _ in range(5):
    response = llm.generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

# Take majority vote
final_answer = most_common(responses)
```

**When to use:** Critical decisions, math problems, complex reasoning.

---

### 7. ReAct (Reason + Act)

Combine reasoning with actions (tool use).

```
Prompt: "Answer the following question using the search tool when needed.

Question: What is the population of the capital of France?

Thought: I need to find the capital of France, then its population.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need to find Paris's population.
Action: search("population of Paris 2024")
Observation: Paris has a population of approximately 2.1 million.

Thought: I have the answer.
Answer: The population of Paris, the capital of France, is approximately 2.1 million."
```

---

### 8. Tree of Thoughts (ToT)

Explore multiple reasoning paths, evaluate, and backtrack.

```
Problem: "24 Game: Use 4, 6, 7, 8 with +, -, *, / to make 24"

Path 1: 4 + 6 = 10, 10 + 7 = 17, 17 + 8 = 25 ❌
Path 2: 8 - 4 = 4, 4 * 6 = 24 ✓ (but didn't use 7) ❌
Path 3: 6 - 4 = 2, 7 + 8 = 15, won't work ❌
Path 4: 8 * (7 - 4 - 6/6)... exploring...
Path 5: (8 - 4) * 6 * (7 - 7)... no
Path 6: 4 * 6 = 24, need to neutralize 7 and 8: 7 - 7 + 8 - 8 = 0
        But can't use same number twice...
Path 7: (8 - 6) * (4 + 7) + 2 = 24? No, no 2...
Path 8: 8 / (4 - 6/7) -- wait, wrong
Path 9: (7 - 4) * 8 = 24, still need 6: 24 * 6/6 = 24 ✓

Answer: (7 - 4) * 8 * (6/6) = 24
```

---

### 9. Role Prompting

Assign a persona to the model.

```
Prompt: "You are a senior Python developer with 15 years of experience
in building scalable systems. Review this code for potential issues:

[code here]"
```

**Effective roles:**
- Expert personas: "You are a Stanford professor of..."
- Specific skills: "You are a technical writer who..."
- Perspective: "You are a skeptical reviewer who..."

---

### 10. Structured Output Prompting

Request specific output formats.

```
Prompt: "Analyze this product review and return JSON:

Review: 'Great laptop but battery life could be better'

Return format:
{
  \"sentiment\": \"positive/negative/mixed\",
  \"aspects\": [{\"aspect\": \"...\", \"sentiment\": \"...\"}],
  \"summary\": \"one sentence summary\"
}"

Output:
{
  "sentiment": "mixed",
  "aspects": [
    {"aspect": "overall quality", "sentiment": "positive"},
    {"aspect": "battery life", "sentiment": "negative"}
  ],
  "summary": "Positive review of laptop with criticism of battery life"
}
```

---

### 11. Delimiter Usage

Use clear delimiters to separate sections.

```
Prompt: "Summarize the text between triple backticks:

```
Long article text here that needs to be summarized.
Multiple paragraphs of content...
```

Provide a 2-sentence summary."
```

**Common delimiters:**
- Triple backticks: \`\`\`text\`\`\`
- Triple quotes: """text"""
- XML tags: <text>content</text>
- Brackets: [text]
- Dashes: ---text---

---

### 12. Instruction Hierarchy

Structure complex prompts clearly.

```
Prompt: "# Task
Analyze the customer feedback below.

# Instructions
1. Identify the main complaint
2. Classify severity (low/medium/high)
3. Suggest a response

# Customer Feedback
\"I've been waiting 3 weeks for my refund!\"

# Output Format
- Complaint:
- Severity:
- Suggested Response:"
```

---

## Advanced Techniques

### 13. Meta-Prompting

Ask the model to improve its own prompt.

```
Prompt: "I want to create a prompt that helps generate creative
product names. Here's my current prompt:

'Give me product names for a water bottle'

Improve this prompt to get better, more creative results."

Output: "Here's an improved prompt:
'Generate 5 creative product names for an eco-friendly,
insulated water bottle targeting fitness enthusiasts.
For each name, provide:
1. The name
2. The feeling/emotion it evokes
3. Target demographic appeal

Style: Modern, energetic, memorable'"
```

---

### 14. Prompt Chaining

Break complex tasks into steps.

```python
# Step 1: Extract key information
extract_prompt = "Extract key facts from this article: {article}"
facts = llm(extract_prompt)

# Step 2: Generate questions
question_prompt = "Generate 5 quiz questions from these facts: {facts}"
questions = llm(question_prompt)

# Step 3: Create answers
answer_prompt = "Provide answers for these questions: {questions}"
answers = llm(answer_prompt)
```

---

### 15. Negative Prompting

Tell the model what NOT to do.

```
Prompt: "Write a product description for headphones.

DO NOT:
- Use clichés like 'crystal clear sound'
- Include technical specifications
- Use more than 50 words
- Use exclamation marks

DO:
- Focus on emotional benefits
- Use sensory language
- Sound premium and sophisticated"
```

---

### 16. Contextual Compression

For long contexts, compress before using.

```
Prompt 1: "Summarize the key points from this document relevant
to the question 'What are the refund policies?'

Document: [long document]"

Prompt 2: "Based on this context, answer the question:

Context: [compressed summary]
Question: What are the refund policies?"
```

---

## Prompt Templates

### Classification Template
```
Classify the following [ITEM_TYPE] into one of these categories:
[CATEGORY_LIST]

[ITEM_TYPE]: [INPUT]

Return only the category name.
```

### Extraction Template
```
Extract the following information from the text:
- [FIELD_1]
- [FIELD_2]
- [FIELD_3]

Text: [INPUT]

Return as JSON.
```

### Summarization Template
```
Summarize the following [CONTENT_TYPE] in [LENGTH] sentences.
Focus on [FOCUS_AREA].
Target audience: [AUDIENCE].

[CONTENT_TYPE]:
[INPUT]

Summary:
```

### Code Generation Template
```
Write a [LANGUAGE] function that:
- Input: [INPUT_DESCRIPTION]
- Output: [OUTPUT_DESCRIPTION]
- Requirements: [REQUIREMENTS]

Include:
- Type hints
- Docstring
- Error handling
- Example usage
```

### Analysis Template
```
Analyze the following [ITEM] considering:
1. [ASPECT_1]
2. [ASPECT_2]
3. [ASPECT_3]

[ITEM]:
[INPUT]

Provide your analysis in the following format:
- [ASPECT_1]: [Analysis]
- [ASPECT_2]: [Analysis]
- [ASPECT_3]: [Analysis]
- Overall Assessment: [Summary]
```

---

## Best Practices

### Do's ✓

1. **Be explicit about format**
   ```
   "Return your answer as a bulleted list with exactly 5 items"
   ```

2. **Specify length**
   ```
   "In 2-3 sentences..." or "In approximately 200 words..."
   ```

3. **Give examples for complex tasks**
   ```
   "Format like this example: [example]"
   ```

4. **Use system prompts for consistent behavior**
   ```
   System: "You are a helpful coding assistant. Always provide
   code examples in Python unless otherwise specified."
   ```

5. **Iterate and refine**
   ```
   Version 1 → Test → Version 2 → Test → Version 3
   ```

### Don'ts ✗

1. **Don't be vague**
   ```
   ❌ "Make it better"
   ✓ "Improve readability by adding section headers and bullet points"
   ```

2. **Don't overload with instructions**
   ```
   ❌ 20 different requirements in one prompt
   ✓ Break into multiple prompts or prioritize key requirements
   ```

3. **Don't assume context**
   ```
   ❌ "Fix the bug" (what bug?)
   ✓ "Fix the null pointer exception on line 15"
   ```

4. **Don't use ambiguous pronouns**
   ```
   ❌ "Compare it to the other one"
   ✓ "Compare Product A to Product B"
   ```

---

## Temperature & Parameters

### Temperature
```
0.0 - Deterministic, same output every time
    Best for: Facts, code, classification

0.3-0.7 - Balanced creativity
    Best for: General tasks, explanations

0.8-1.0 - More random, creative
    Best for: Creative writing, brainstorming

1.0+ - Very random
    Best for: Poetry, experimental content
```

### Top-P (Nucleus Sampling)
```
0.1 - Very focused (top 10% probability mass)
0.5 - Moderate diversity
0.9 - High diversity (default for most tasks)
1.0 - Consider all tokens
```

### Recommended Settings

| Task | Temperature | Top-P |
|------|-------------|-------|
| Code generation | 0.0-0.2 | 0.1 |
| Factual Q&A | 0.0 | 0.1 |
| Classification | 0.0 | 0.1 |
| Summarization | 0.3 | 0.5 |
| Conversation | 0.7 | 0.9 |
| Creative writing | 0.9 | 0.95 |
| Brainstorming | 1.0 | 1.0 |

---

## Common Patterns

### The CRISPE Framework
```
C - Capacity: What role should the AI assume?
R - Request: What do you want?
I - Instructions: How should it be done?
S - Style: What tone/format?
P - Personality: Any character traits?
E - Examples: Sample outputs?
```

### The RACE Framework
```
R - Role: "You are a..."
A - Action: "Your task is to..."
C - Context: "Given that..."
E - Examples: "For instance..."
```

### The CREATE Framework
```
C - Character: Who is the AI?
R - Request: What to do?
E - Examples: Show samples
A - Adjustments: Constraints
T - Type: Output format
E - Extras: Additional notes
```

---

## Interview Questions

**Q: What's the difference between few-shot and fine-tuning?**
> Few-shot: Examples in prompt, no model change, flexible, limited by context
> Fine-tuning: Train on examples, permanent model change, better for specific tasks

**Q: When would you use Chain-of-Thought?**
> Math problems, logic puzzles, multi-step reasoning, when you need to verify the process, complex analysis

**Q: How do you reduce hallucinations through prompting?**
> 1) Ask for sources/citations
> 2) "If unsure, say 'I don't know'"
> 3) Lower temperature
> 4) Ground in provided context (RAG)
> 5) Ask for confidence levels

**Q: How do you test prompt quality?**
> 1) Test on diverse inputs
> 2) Check edge cases
> 3) Measure consistency (same input = similar output)
> 4) A/B test with users
> 5) Use evaluation frameworks (human + automated)
