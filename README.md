# LLM Generated RAG Code
The code in this repo is provided by different LLMs when prompted to provide Python code to do Retrieval Augemented Generation using the OpenAI API.

The repo contains at least 2 branches. 

The main branch is the LLM Generated code for performing RAG against a single PDF file. The other branch contains the generated code with edits to make it runnable with limited change in the generated code. Comments are addeded to the top of the file to indicate if the generated code has been edited to provide a functioning set of code.
Not all generated code was edited to make it functional - for instance the CoPilot generated code was so far out of bounds from conceptually working there was no mechanism to make it conceptually functional. The code could have been made to run but it woud not have remotely worked as intended thus it was not edited.

The primary edits were to make it function against the latest, at the time, Python OpenAI package. This is because the generated code first created in September of 2024 used a significantly older OpenAI package. This is understandable based on the information up to the date the models would have been trained on. In general the edits consist of:
* Update calls to use the proper OpenAI 1.0+ package functions, paramaters, etc.
* Set the models to be used to something more current than generated
* Load of the OPENAI_API_KEY into the needed variable
* Change the basic query for the prompt
* Point to the desired PDF file for RAG use

The PDF document used is not small. It's 43,000+ tokens depending on how you tokenize it. This is why you see gpt-4o-mini model used; it was essential in order to get the throughput of the entire document in context in the way in which the Cursor generated code met the challenge. The gpt-4o-mini model allows for 300,000 tokens per minute on low tiers.

There are not specific metrics on the cost of each run but I am fairly certain it was significantly below $0.01(USD) per execution. Perhaps these values can be gathered soon to ensure any person playing with the code can be aware of the per complete run cost.

Due to the nature of the 2+ branches and what that represent you can't think of this repo like a standard repo with 2 branches that you'd perform merges between them. If more generated code from other LLMs is added to the repo's main branch then that commit will be cherry picked into the branch with edits or a duplicate commit will be created in the branch containing edits. The main branch will never be merged into the edited generated code branch(es) and likewise between the edited code and main.

So that one does not have to actually run the code at a cost, I tried to put in the comments, on the exectuable code, the final response and where appropriate and within reason the chunk(s) passed to the LLM for a context in yielding a final response. 
