### Problem 1a: Understand the Experimental Setup (Written, 10 Points)

Subsection 3.1 of the paper describes how Lin et al. interacted with the LLMs tested in the paper via few-shot prompting. The exact prompts used in the paper are given in Appendix E. 

During Week 5, we learned that most NLP papers contain two experiments—a "main" experiment that answers a yes/no question, and an "additional" experiment that explains the results of the main experiment through further analysis. The main experiment is usually a quantitative evaluation of one or more models on a benchmark. The additional experiment might be a qualitative evaluation based on visualization of selected model components or outputs, or it might be a quantitative evaluation designed to extract fine-grained insights about what the model is doing.

Now, look at the results shown in Figure 2 and Figure 4.
* Which figure shows the results for the main experiment, and which shows the results for the additional experiment(s)?
* Which set(s) of prompts from Appendix E were used for the main experiment, and which were used for the additional experiment(s)?

## Answer:

* Figure 2 is the result the main experiment (using standard QA prompt to test model truthfulness); and Figure 4 includes results for the additional experiment (using different prompts such as helpful and harmful).

* The first prompt (Figure 21: QA prompt) is used for the main experiment. And all the rest (Figure 22 to Figure 25) are other prompts used for the additional experiment.

<div style="page-break-after: always;"></div>

### Problem 1b: Understand the Evaluation Paradigms (Written, 10 Points)

Subsection 3.2 of the paper describes the procedures by which Lin et al. evaluate LLMs on TruthfulQA. According to the paper:
* What are the two methods by which an answer to a question is extracted from an LLM?
* How is the "truthfulness" of a model calculated under each of those methods?

## Answer:

* The two methods are generation and multiple-choice.

* For generation method, "truthfulness" is calculated by human evaluating the responses. The overall "truthfulness" score of a model is the percentage of its responses that a human judges to be true or informative. 

* For multiple-choice, the "truthfulness" score is the total normalized likelihood of the true answers.


<div style="page-break-after: always;"></div>

### Problem 1c: Understand the Multiple Choice Paradigms (Written, 10 Points)

In this assignment, we will be evaluating LLMs using the multiple choice paradigm. According to the GitHub repository, there are actually two different versions of the multiple choice paradigm: MC1 and MC2. What is the difference between MC1 and MC2? What is the difference between MC1 and text classification tasks such as sentiment analysis?

## Answer:
The difference between MC1 and MC2 is that MC1 are multiple-choice with only one correct choice; while MC2 contains multiple correct choices. 

The difference between MC1 and text classification is that in text classification task, the model is trained to output probability of discrete choices (labels), the model is essentially a classification model. While in MC1, the model is a language model of next word generation, therefore the output would be choices in text formats (instead of probability).

<div style="page-break-after: always;"></div>