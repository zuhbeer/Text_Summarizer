Proposal
===========

I. Objective
---------

Build a pipeline that can take a sentence, paragraph, or whole document and summarize it into coherant English. This is an incredibly difficult task that is currently being researched. If I cannot create a summarizer I will at least attempt to extract only the most important information from blocks of text and present it coherantly.
  
  
II. Existing Solutions
----------

This problem is not 100% solved, but there are amazing technologies and companies working to solve this such as Google (BERT), IBM (MAX), and ELMo. I have also noticed bots on Reddit, specifically a bot named 'autotldr', which utilizes the SMMRY algorithm below to summarize news articles linked on the page.

- Ranking sentences by importance using the core algorithm:
  1) Associate words with their grammatical counterparts. (e.g. "city" and "cities")
  2) Calculate the occurrence of each word in the text.
  3) Assign each word with points depending on their popularity.
  4) Detect which periods represent the end of a sentence. (e.g "Mr." does not).
  5) Split up the text into individual sentences.
  6) Rank sentences by the sum of their words' points.
  7) Return X of the most highly ranked sentences in chronological order

- Reorganizing the summary to focus on a topic; by selection of a keyword
- Removing transition phrases
- Removing unnecessary clauses
- Removing excessive examples
  

III. Novel Approach
------------

I think I can build on IBM or Google's work and make use of some aspects of their models to build a better summarizer. I would like to look through their code and tune their models where I can to give me a better understanding of NLP, RNN's, and transfer learning. I expect my hypothesis or approach to evolve somewhat throughout the process.
  

IV. Impact
------------

A good summarizer can be used everyday to stay up to date with current events, read articles faster, or simplify lengthy documents. The importance of a summarizer is dependent on its accuracy in conveying the true meaning of a paragraph or article, which is a daunting task. For this capstone, any step towards a "meaningful" summarizer would satisfy me. However, a project like this could never be 100% complete as colloquial language is always changing. And tuning a model to simplify English into a most semantically basic form could take years.

V. Presentation
-----------

I'd like to present my work as a flask app or browser extension to summarize any input or highlighted text. The app should be able take plain text as an input but ultimately take urls and uploaded files.

VI. Data Source
-----------

I currently have a dataset of legal documents and their summaries, as well as tagged science articles and summaries. This data is mostly zipped xml files. I would also like to train my model on novels and their respective cliff notes.

VII. Potential Problems
------------

There are many issues with this project, as different types of data (news, science, literature) have varying levels of subjectivity and density, and different syntax and semantics. Getting a computer to understand semantics is seemingly impossible, but with enough training data the computer wouldn't have to actually understand meaning. Additionally, this project may require spinning up a machine on AWS which is another hurdle in the pipeline.

VIII. Next Steps
------------

For a better model, the first step would be to acquire more datasets and then start training RNN models on the data.

-------------
*Sources*

https://github.com/IBM/MAX-Text-Summarizer

https://smmry.com/

https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/

-------------
*Dataset*

Amazon food reviews & summaries - https://www.kaggle.com/snap/amazon-fine-food-reviews

legal documents and their summaries - https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports

tagged science articles and summaries - https://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html

CNN / Daily mail datasets - https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

Novels and cliffnotes/sparknotes summaries -

