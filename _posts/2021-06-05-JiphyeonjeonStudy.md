---
title: "집현전 스터디 2기 최신반 3조 논문 리스트"
date: 2021-06-05 23:15:28 -0400
categories: NLP
tags:
  - Paper review
  - Deep Learning
  - NLP
  - Dialogue System

use_math: false
toc: false
---

# 집현전 스터디 2기 최신반 3조 논문 리스트

## 배경 지식

1. [[chatbot] 챗봇의 동향, 기술... 등에 대하여(1)](https://kmkidea.tistory.com/47)

1. [Chatbots & Dialogue Systems](http://web.stanford.edu/~jurafsky/slp3/24.pdf)

## EMNLP 2020

1. Multi-turn Response Selection using Dialogue Dependency Relation
    * Abstract : Multi-turn response selection is a task designed for developing dialogue agents. The performance on this task has a remarkable improvement with pre-trained language models. However, these models simply concatenate the turns in dialogue history as the input and largely ignore the dependencies between the turns. In this paper, we propose a dialogue extraction algorithm to transform a dialogue history into threads based on their dependency relations. Each thread can be regarded as a self-contained sub-dialogue. We also propose Thread-Encoder model to encode threads and candidates into compact representations by pre-trained Transformers and finally get the matching score through an attention layer. The experiments show that dependency relations are helpful for dialogue context understanding, and our model outperforms the state-of-the-art baselines on both DSTC7 and DSTC8*, with competitive results on UbuntuV2.
    * Affiliation : Shanghai Jiao Tong University
    * Link : https://arxiv.org/pdf/2010.01502

1. The World is Not Binary: Learning to Rank with Grayscale Data for Dialogue Response Selection
    * Abstract : Response selection plays a vital role in building retrieval-based conversation systems. Despite that response selection is naturally a learning-to-rank problem, most prior works take a point-wise view and train binary classifiers for this task: each response candidate is labeled either relevant (one) or irrelevant (zero). On the one hand, this formalization can be sub-optimal due to its ignorance of the diversity of response quality. On the other hand, annotating grayscale data for learning-to-rank can be prohibitively expensive and challenging. In this work, we show that grayscale data can be automatically constructed without human effort. Our method employs off-the-shelf response retrieval models and response generation models as automatic grayscale data generators. With the constructed grayscale data, we propose multi-level ranking objectives for training, which can (1) teach a matching model to capture more fine-grained context-response relevance difference and (2) reduce the train-test discrepancy in terms of distractor strength. Our method is simple, effective, and universal. Experiments on three benchmark datasets and four state-of-the-art matching models show that the proposed approach brings significant and consistent performance improvements.
    * Affiliation : Tsinghua Shenzhen International Graduate School, Tsinghua University
    * Link : https://arxiv.org/pdf/2004.02421

1. Proﬁle Consistency Identiﬁcation for Open-domain Dialogue Agents
    * Abstract : Maintaining a consistent attribute profile is crucial for dialogue agents to naturally converse with humans. Existing studies on improving attribute consistency mainly explored how to incorporate attribute information in the responses, but few efforts have been made to identify the consistency relations between response and attribute profile. To facilitate the study of profile consistency identification, we create a large-scale human-annotated dataset with over 110K single-turn conversations and their key-value attribute profiles. Explicit relation between response and profile is manually labeled. We also propose a key-value structure information enriched BERT model to identify the profile consistency, and it gained improvements over strong baselines. Further evaluations on downstream tasks demonstrate that the profile consistency identification model is conducive for improving dialogue consistency.
    * Affiliation : Research Center for Social Computing and Information Retrieval
Harbin Institute of Technology, Heilongjiang, China
    * Link : https://arxiv.org/pdf/2009.09680

1. Knowledge-Grounded Dialogue Generation with Pre-trained Language Models
    * Abstract : We study knowledge-grounded dialogue generation with pre-trained language models. To leverage the redundant external knowledge under capacity constraint, we propose equipping response generation defined by a pre-trained language model with a knowledge selection module, and an unsupervised approach to jointly optimizing knowledge selection and response generation with unlabeled dialogues. Empirical results on two benchmarks indicate that our model can significantly outperform state-of-the-art methods in both automatic evaluation and human judgment.
    * Affiliation : Wangxuan Institute of Computer Technology, Peking University, Beijing, China
    * Link : https://arxiv.org/pdf/2010.08824

1. MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems
    * Abstract : In this paper, we propose Minimalist Transfer Learning (MinTL) to simplify the system design process of task-oriented dialogue systems and alleviate the over-dependency on annotated data. MinTL is a simple yet effective transfer learning framework, which allows us to plug-and-play pre-trained seq2seq models, and jointly learn dialogue state tracking and dialogue response generation. Unlike previous approaches, which use a copy mechanism to "carryover" the old dialogue states to the new one, we introduce Levenshtein belief spans (Lev), that allows efficient dialogue state tracking with a minimal generation length. We instantiate our learning framework with two pre-trained backbones: T5 and BART, and evaluate them on MultiWOZ. Extensive experiments demonstrate that: 1) our systems establish new state-of-the-art results on end-to-end response generation, 2) MinTL-based systems are more robust than baseline methods in the low resource setting, and they achieve competitive results with only 20\% training data, and 3) Lev greatly improves the inference efficiency.
    * Affiliation : Center for Artificial Intelligence Research (CAiRE)
    * Link : https://arxiv.org/pdf/2009.12005

1. TOD-BERT:Pre-trained Natural Language Understanding for Task-Oriented Dialogue
    * Abstract : The underlying difference of linguistic patterns between general text and task-oriented dialogue makes existing pre-trained language models less useful in practice. In this work, we unify nine human-human and multi-turn task-oriented dialogue datasets for language modeling. To better model dialogue behavior during pre-training, we incorporate user and system tokens into the masked language modeling. We propose a contrastive objective function to simulate the response selection task. Our pre-trained task-oriented dialogue BERT (TOD-BERT) outperforms strong baselines like BERT on four downstream task-oriented dialogue applications, including intention recognition, dialogue state tracking, dialogue act prediction, and response selection. We also show that TOD-BERT has a stronger few-shot ability that can mitigate the data scarcity problem for task-oriented dialogue.
    * Affiliation : Salesforce Research
    * Link : https://arxiv.org/pdf/2004.06871

## NAACL 2021

1. Fine-grained Post-training for Improving Retrieval-based Dialogue
Systems
    * Abstract : Retrieval-based dialogue systems display an outstanding performance when pre-trained language models are used, which includes bidirectional encoder representations from transformers (BERT). During the multi-turn response selection, BERT focuses on training the relationship between the context with multiple utterances and the response. However, this method of training is insufficient when considering the relations between each utterance in the context. This leads to a problem of not completely understanding the context flow that is required to select a response. To address this issue, we propose a new fine-grained post-training method that reflects the characteristics of the multi-turn dialogue. Specifically, the model learns the utterance level interactions by training every short context-response pair in a dialogue session. Furthermore, by using a new training objective, the utterance relevance classification, the model understands the semantic relevance and coherence between the dialogue utterances. Experimental results show that our model achieves new state-of-the-art with significant margins on three benchmark datasets. This suggests that the fine-grained post-training method is highly effective for the response selection task.
    * Affiliation : Department of Computer Science and Engineering, Sogang University
    * Link : https://www.aclweb.org/anthology/2021.naacl-main.122.pdf

1. Bot-Adversarial Dialogue for Safe Conversational Agents
    * Abstract : Conversational agents trained on large unlabeled corpora of human interactions will learn patterns and mimic behaviors therein, which include offensive or otherwise toxic behavior. We introduce a new human-and-model-in-the-loop framework for evaluating the toxicity of such models, and compare a variety of existing methods in both the cases of non-adversarial and adversarial users that expose their weaknesses. We then go on to propose two novel methods for safe conversational agents, by either training on data from our new human-and-model-in-the-loop framework in a two-stage system, or ”baking-in” safety to the generative model itself. We find our new techniques are (i) safer than existing models; while (ii) maintaining usability metrics such as engagingness relative to state-of-the-art chatbots. In contrast, we expose serious safety issues in existing standard systems like GPT2, DialoGPT, and BlenderBot.
    * Affiliation : Facebook AI Research
    * Link : https://www.aclweb.org/anthology/2021.naacl-main.235.pdf

1. Imperfect also Deserves Reward: Multi-Level and Sequential Reward
Modeling for Better Dialog Management
    * Abstract : For task-oriented dialog systems, training a Reinforcement Learning (RL) based Dialog Management module suffers from low sample efficiency and slow convergence speed due to the sparse rewards in this http URL solve this problem, many strategies have been proposed to give proper rewards when training RL, but their rewards lack interpretability and cannot accurately estimate the distribution of state-action pairs in real dialogs. In this paper, we propose a multi-level reward modeling approach that factorizes a reward into a three-level hierarchy: domain, act, and slot. Based on inverse adversarial reinforcement learning, our designed reward model can provide more accurate and explainable reward signals for state-action pairs.Extensive evaluations show that our approach can be applied to a wide range of reinforcement learning-based dialog systems and significantly improves both the performance and the speed of convergence.
    * Affiliation : Tencent Jarvis Lab
    * Link : https://arxiv.org/pdf/2104.04748

## ACL 2020
1. Efficient Dialogue State Tracking by Selectively Overwriting Memory
    * Abstract : Recent works in dialogue state tracking (DST) focus on an open vocabulary-based setting to resolve scalability and generalization issues of the predefined ontology-based approaches. However, they are inefficient in that they predict the dialogue state at every turn from scratch. Here, we consider dialogue state as an explicit fixed-sized memory and propose a selectively overwriting mechanism for more efficient DST. This mechanism consists of two steps: (1) predicting state operation on each of the memory slots, and (2) overwriting the memory with new values, of which only a few are generated according to the predicted state operations. Our method decomposes DST into two sub-tasks and guides the decoder to focus only on one of the tasks, thus reducing the burden of the decoder. This enhances the effectiveness of training and DST performance. Our SOM-DST (Selectively Overwriting Memory for Dialogue State Tracking) model achieves state-of-the-art joint goal accuracy with 51.72% in MultiWOZ 2.0 and 53.01% in MultiWOZ 2.1 in an open vocabulary-based DST setting. In addition, we analyze the accuracy gaps between the current and the ground truth-given situations and suggest that it is a promising direction to improve state operation prediction to boost the DST performance.
    * Affiliation : Clova AI, NAVER Corp.
    * Link : https://arxiv.org/pdf/1911.03906

1. Large Scale Multi-Actor Generative Dialog Modeling
    * Abstract : Non-goal oriented dialog agents (i.e. chatbots) aim to produce varying and engaging conversations with a user; however, they typically exhibit either inconsistent personality across conversations or the average personality of all users. This paper addresses these issues by controlling an agent's persona upon generation via conditioning on prior conversations of a target actor. In doing so, we are able to utilize more abstract patterns within a person's speech and better emulate them in generated responses. This work introduces the Generative Conversation Control model, an augmented and fine-tuned GPT-2 language model that conditions on past reference conversations to probabilistically model multi-turn conversations in the actor's persona. We introduce an accompanying data collection procedure to obtain 10.3M conversations from 6 months worth of Reddit comments. We demonstrate that scaling model sizes from 117M to 8.3B parameters yields an improvement from 23.14 to 13.14 perplexity on 1.7M held out Reddit conversations. Increasing model scale yielded similar improvements in human evaluations that measure preference of model samples to the held out target distribution in terms of realism (31% increased to 37% preference), style matching (37% to 42%), grammar and content quality (29% to 42%), and conversation coherency (32% to 40%). We find that conditionally modeling past conversations improves perplexity by 0.47 in automatic evaluations. Through human trials we identify positive trends between conditional modeling and style matching and outline steps to further improve persona control.
    * Affiliation : NVIDIA
    * Link : https://arxiv.org/pdf/2005.06114

1. USR: An Unsupervised and Reference Free Evaluation Metric for Dialog
Generation
    * Abstract : The lack of meaningful automatic evaluation metrics for dialog has impeded open-domain dialog research. Standard language generation metrics have been shown to be ineffective for evaluating dialog models. To this end, this paper presents USR, an UnSupervised and Reference-free evaluation metric for dialog. USR is a reference-free metric that trains unsupervised models to measure several desirable qualities of dialog. USR is shown to strongly correlate with human judgment on both Topical-Chat (turn-level: 0.42, system-level: 1.0) and PersonaChat (turn-level: 0.48 and system-level: 1.0). USR additionally produces interpretable measures for several desirable properties of dialog.
    * Affiliation : Dialog Research Center, Language Technologies Institute, Carnegie Mellon University, USA
    * Link : https://www.aclweb.org/anthology/2020.acl-main.64.pdf
