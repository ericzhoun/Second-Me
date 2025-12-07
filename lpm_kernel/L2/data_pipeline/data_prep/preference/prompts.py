CH_USR_TEMPLATES = {
    "query": """请根据要求，站在用户视角，提出一个关于我（用户）个人兴趣的问题，以下是参考信息：

{bio}

## 用户记录在知识库中的内容 ##

{chunks_concat}""",
    "answer": """以下是从用户角度提出的问题，请你站在用户的个人AI助手的角度，根据用户画像和用户记录在知识库中的内容生成答案：
{question}

以下是用户画像：

{bio}

以下是用户记录在知识库中的内容：

{chunks_concat}""",
}


EN_USR_TEMPLATES = {
    "query": """Please propose a question about my (the user’s) personal interests from the user’s perspective, based on the following reference information:

{bio}

## User Records in the Knowledge Base ##

{chunks_concat}""",
    "answer": """Here is the question posed from the user’s perspective. Please generate an answer from the standpoint of the user’s personal AI assistant, using the user profile and content recorded in the knowledge base:
{question}

Here is the user profile:

{bio}

Here is the content the user has read in the knowledge base:

{chunks_concat}""",
}


CH_SYS_TEMPLATES = {
    "query": """# Role #
你是一个优秀的文字工作者，你擅长于根据用户画像和用户记录在知识库中的内容，生成有关用户行为预测的问题。

# Goal #
你将会读到用户画像和用户记录在知识库中的内容，根据用户画像从用户记录中选择有价值的信息，站在用户的角度，提出一个关于用户行为预测的问题。

# Guidelines #
1. 请根据用户记录在知识库中的内容的信息量决定你生成问题的复杂度。
2. 你的问题应该足够有指向性，保证生成的问题是和用户以及用户记录在知识库中的内容里提及的实体相关的。生成的问题中应该包含用户提供的实体和具体情境、事件、地点等。
3. 请确保生成的问题是有意义的，不要生成无意义的问题，也不要生成以用户画像为答案的问题。
4. 你只需要在你的回复中包含你生成的问题，不需要包含其他信息。
5. 确保你提问的视角是站在用户的角度，以“我”为主体提问。
""",
    "answer": """# Role #
你是一个优秀的文字工作者，你擅长于根据用户画像和所读内容，以及一个从用户角度提出的问题，生成有关用户行为预测的答案。

# Goal #
你将会读到用户画像和用户记录在知识库中的内容，以及用户提出的问题，根据用户画像和用户记录中的信息，站在用户的角度，回答用户提出的问题。

# Guidelines #
1. 请根据用户记录在知识库中的内容的信息量，以及问题的复杂度，决定你生成答案的复杂度。
2. 你回答的内容应该足够有组织性，保证回答的内容是和用户以及用户记录在知识库中的内容里提及的实体相关的。回答的内容中应该包含用户提供的实体和具体情境、事件、地点等。
3. 请确保回答的内容主要来自用户记录在知识库中的内容，而不是照抄用户画像作为答案。
4. 你只需要在你的回复中包含你生成的答案，不需要包含其他信息。
5. 确保你回答的视角是站在用户的个人AI助手的角度。
""",
}


EN_SYS_TEMPLATES = {
    "query": """# Role #
You are an excellent writer, you are good at generating questions about user personal interests based on the user profile and the content they have read.

# Goal #
You will read the user profile and the content the user has read in the knowledge base, select valuable information from the user profile, and generate a question about user personal interests based on the user profile.

# Guidelines #
1. Please determine the complexity of the question you generate based on the information in the user profile.
2. Your question should be specific enough to ensure that the generated question is relevant to the entities mentioned in the user profile and the specific context, events, locations, etc. mentioned in the user's records.
3. Please ensure that the generated question is meaningful and not meaningless, and does not generate questions with the user profile as the answer.
4. You only need to include the question you generated in your response, no other information is required.
5. Ensure that the perspective of your question is from the user's perspective, asking questions with "I" as the subject.
""",
    "answer": """# Role #
You are an excellent writer, you are good at generating answers about user personal interests based on the user profile and the content they have read.

# Goal #
You will read the user profile and the content the user has read in the knowledge base, as well as a question asked by the user, and generate an answer about user personal interests based on the user profile.

# Guidelines #
1. Please determine the complexity of the answer you generate based on the information in the user profile and the complexity of the question.
2. Your answer should be organized enough to ensure that the answer is relevant to the entities mentioned in the user profile and the specific context, events, locations, etc. mentioned in the user's records.
3. Please ensure that the answer is mainly derived from the content the user has read in the knowledge base, rather than copying the user profile as the answer.
4. You only need to include the answer you generated in your response, no other information is required.
5. Ensure that the perspective of your answer is from the standpoint of the user's personal AI assistant.
""",
}


CH_USR_COT_TEMPLATES = {
    "query": """请根据要求，站在用户视角，用中文提出一个关于我（用户）个人兴趣的问题，以下是参考信息：

{bio}

## 用户记录在知识库中的内容 ##

{chunks_concat}

使用以下格式构建你的回复：<question>(生成的问题)</question>
""",
    "answer": """以下是从用户角度提出的问题:
{question}

以下是用户画像：

{bio}

以下是用户记录在知识库中的内容：

{chunks_concat}

请你站在用户的个人AI助手的角度，根据用户画像和用户记录在知识库中的内容用中文生成带推理链的符合要求的格式的答案。请确保以<think>开始你的输出，以</answer>作为你输出的结束。"""
}


EN_USR_COT_TEMPLATES = {
    "query": """Please use English to propose a question about my (the user’s) personal interests from the user’s perspective, based on the following reference information:

{bio}

## User Records in the Knowledge Base ##

{chunks_concat}""",
    "answer": """Here is the question posed from the user’s perspective. Please use English to generate an answer from the standpoint of the user’s personal AI assistant, using the user profile and content recorded in the knowledge base:
{question}

Here is the user profile:

{bio}

Here is the content the user has read in the knowledge base:

{chunks_concat}"""
}


CH_SYS_COT_TEMPLATES = {
    "query": """# Role #
你是一个优秀的文字工作者，你擅长于根据用户画像和用户记录在知识库中的内容，生成有关用户行为预测的问题。

# Goal #
你将会读到用户画像和用户记录在知识库中的内容，根据用户画像从用户记录中选择有价值的信息，站在用户的角度，提出一个关于用户行为预测的问题。

# Guidelines #
1. 请根据用户记录在知识库中的内容的信息量决定你生成问题的复杂度。
2. 你的问题应该足够有指向性，保证生成的问题是和用户以及用户记录在知识库中的内容里提及的实体相关的。生成的问题中应该包含用户提供的实体和具体情境、事件、地点等。
3. 确保你提问的视角是站在用户的角度，以“我”为主体提问。

# Response Format #
使用以下格式构建你的回复："<question>(生成的问题)</question>"
""",
    "answer": """# Role #
你是一个优秀的文字工作者，你擅长于根据用户画像和所读内容，以及一个从用户角度提出的问题，生成有关用户行为预测的答案。

# Goal #
你将会读到用户画像和用户记录在知识库中的内容，以及用户提出的问题，根据用户画像和用户记录中的信息，站在用户的角度，回答用户提出的问题。

# Guidelines #
1. 请根据用户记录在知识库中的内容的信息量，以及问题的复杂度，决定你生成答案的复杂度。答案应采用链式思维（CoT）推理方法构建，首先重述上下文信息进行思考和推理，然后结合你掌握的信息进行回答。
2. 你回答的内容应该足够有组织性，保证回答的内容是和用户以及用户记录在知识库中的内容里提及的实体相关的。回答的内容中应该包含用户提供的实体和具体情境、事件、地点等。
3. 确保你回答的视角是站在用户的个人AI助手的角度。

# Response Format #
以<think>作为回答的开头，</answer>作为回答的结尾，按该形式进行输出："<think>(思考和推理过程)</think><answer>(最终答案)</answer>"
"""
}


EN_SYS_COT_TEMPLATES = {
    "query": """# Role #
You are an excellent writer, you are good at using English to generate questions about user personal interests based on the user profile and the content they have read.

# Goal #
You will read the user profile and the content the user has read in the knowledge base, select valuable information from the user profile, and generate a question about user personal interests based on the user profile.

# Guidelines #
1. Please determine the complexity of the question you generate based on the information in the user profile.
2. Your question should be specific enough to ensure that the generated question is relevant to the entities mentioned in the user profile and the specific context, events, locations, etc. mentioned in the user's records.
3. Please ensure that the generated question is meaningful and not meaningless, and does not generate questions with the user profile as the answer.
4. You only need to include the question you generated in your response, no other information is required.
5. Ensure that the perspective of your question is from the user's perspective, asking questions with "I" as the subject.
"""
    ,"answer": """# Role #
You are an excellent writer, you are good at using English to generate answers about user personal interests based on the user profile and the content they have read.

# Goal #
You will read the user profile and the content the user has read in the knowledge base, as well as a question asked by the user, and generate an answer about user personal interests based on the user profile.

# Guidelines #
1. Please determine the complexity of the answer you generate based on the information in the user profile and the complexity of the question.
2. Your answer should be organized enough to ensure that the answer is relevant to the entities mentioned in the user profile and the specific context, events, locations, etc. mentioned in the user's records.
3. Please ensure that the answer is mainly derived from the content the user has read in the knowledge base, rather than copying the user profile as the answer.
4. You only need to include the answer you generated in your response, no other information is required.
5. Ensure that the perspective of your answer is from the standpoint of the user's personal AI assistant.
"""
}


# =============================================================================
# BATCH PROCESSING TEMPLATES
# These templates are used to process multiple clusters in a single API call
# =============================================================================

EN_BATCH_SYS_TEMPLATE = """# Role #
You are an excellent writer who generates Q&A pairs about user personal interests.

# Goal #
You will receive multiple content clusters. For EACH cluster, generate ONE question-answer pair.

# Guidelines #
1. Generate exactly one Q&A pair per cluster
2. Questions should be from the user's perspective (using "I")
3. Answers should be from an AI assistant's perspective
4. Output MUST be valid JSON array format
5. Each Q&A pair should be relevant to its specific cluster content

# Output Format #
Return a JSON array with one object per cluster:
[
  {"cluster_index": 0, "question": "...", "answer": "..."},
  {"cluster_index": 1, "question": "...", "answer": "..."},
  ...
]
"""

EN_BATCH_USR_TEMPLATE = """Here is the user profile:

{bio}

Below are {num_clusters} content clusters. Generate ONE question-answer pair for EACH cluster.

{clusters_content}

Return your response as a valid JSON array with {num_clusters} Q&A pairs, one for each cluster.
Remember: Output ONLY the JSON array, no other text."""


CH_BATCH_SYS_TEMPLATE = """# 角色 #
你是一个优秀的文字工作者，擅长生成关于用户个人兴趣的问答对。

# 目标 #
你将收到多个内容集群。对于每个集群，生成一个问答对。

# 指南 #
1. 每个集群生成恰好一个问答对
2. 问题应该站在用户的角度（使用"我"）
3. 答案应该站在AI助手的角度
4. 输出必须是有效的JSON数组格式
5. 每个问答对应与其特定集群内容相关

# 输出格式 #
返回一个JSON数组，每个集群对应一个对象：
[
  {"cluster_index": 0, "question": "...", "answer": "..."},
  {"cluster_index": 1, "question": "...", "answer": "..."},
  ...
]
"""

CH_BATCH_USR_TEMPLATE = """以下是用户画像：

{bio}

以下是{num_clusters}个内容集群。为每个集群生成一个问答对。

{clusters_content}

将你的回复作为有效的JSON数组返回，包含{num_clusters}个问答对，每个集群一个。
记住：只输出JSON数组，不要有其他文本。"""