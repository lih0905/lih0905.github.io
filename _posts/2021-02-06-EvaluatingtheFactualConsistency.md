---
title: ""
date: 2021-02-06 23:15:28 -0400
categories: NLP
tags:
  - Paper review
  - Deep Learning
  - NLP
  - Summarization

use_math: true
#toc: false
---

  * 논문 제목 : [Evaluating the Factual Consistency of Abstractive Text Summarization, Kryściński et al. (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.750.pdf) 
  * [구현 코드](https://github.com/salesforce/factCC)
  
## 1. Introduction

현재 요약 분야에서 제기되는 문제점들은 다음과 같다.

* 사실 관계 일관성 등 중요한 정보를 무시하는 불충분한 평가 지표
* 노이즈가 많아서 태스크에 집중하지 못하도록 하는 데이터셋
* 특정 도메인에 치우친 데이터로 인한 훈련 편향

논문은 이 중에서 원본 문서와 생성된 요약 사이의 사실 관계 일관성에 대해 다루고 있다. 최근 연구에 따르면 추상 요약 모델이 생성한 요약문의 약 30% 가 사실 관계에서 일치하지 않았으며, 이로 인해 요약 모델의 실제 활용에 큰 제약으로 나타나고 있다.

저자진은 BERT를 기반으로 사실 관계 일관성을 판단하는 weakly-supervised 모델을 제안한다. 훈련 데이터는 원본 문서에 규칙 기반 변환을 여러번 적용하여 생성하였다.

## 2. Related Work

* Goodrich et al. (2019) : 사실 관계 정확도를 원본 문서와 요약문에서 추출한 사실들 사이의 precision으로 정의함. 긍정적인 결과를 얻었으나, 부정문이나 유의어 등의 관계를 파악하지 못하는 단점이 있음

* Falke et al. (2019) : 요약문 생성에 beam search를 사용하여 여러 후보군을 생성한 뒤 요약문별 정확도 점수를 산출하여 가장 높은 후보를 반환함

* Cao et al. (2018) : 듀얼 인코더를 사용, 각각 본문과 포함된 사실들을 인코딩한다. 생성 과정에서 디코더는 인코딩된 본문과 사실들 모두에 attend하여 사실들을 담아낼 수 있도록 강제한다.


## 3. Methods

SOTA급 요약 모델들에서 발생하는 사실 관계 불일치 오류들을 분석하며 알게된 것들은 다음과 같다.

* 