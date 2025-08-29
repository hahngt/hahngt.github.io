---
layout: post
title: Daily Papers — 2025-08-29"
date: 2025-08-29 08:15:00
tags: [papers, arxiv, ai]
---

### 1. [Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable

Text-to-Image Reinforcement Learning](https://arxiv.org/abs/2508.20751)

제목: Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable  
Text-to-Image Reinforcement Learning  
arXiv: https://arxiv.org/abs/2508.20751

## Introduction

- Goal: 본 연구는 안정적인 텍스트-투-이미지(T2I) 강화학습을 위해 기존 점수 기반 보상 모델의 불안정성을 극복하는 선호(pairwise preference) 기반 GRPO 방법을 제안하는 데 목적이 있다.
- Motivation: 기존 점수 정규화 방식은 유사한 보상 점수들 간 차이가 작아 불합리하게 증폭된 이른바 환상적 이점(illusory advantage)이 발생하여 보상 해킹 문제를 초래한다는 한계가 존재한다.
- Contribution: 본 논문은 환상적 이점 문제를 분석하고 이를 해결하기 위해 이미지 쌍 간 선호 관계를 활용하는 PREF-GRPO를 제안하며, 이를 뒷받침하는 세밀한 평가를 위한 통합 벤치마크 UNIGENBENCH를 함께 제시한다.

## Method

PREF-GRPO는 기존 절대 점수 최대화 대신 이미지 그룹 내 모든 쌍에 대해 선호 보상 모델로 상대적 우위를 평가하고, 각 이미지의 승률을 정책 최적화의 보상 신호로 사용한다.

### 2. [rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722)

제목: rStar2-Agent: Agentic Reasoning Technical Report  
arXiv: https://arxiv.org/abs/2508.20722

## Introduction

- Goal: 본 연구의 목표는 agentic 강화학습을 통해 14B 규모의 수학 추론 모델 rStar2-Agent를 개발하여 최첨단 수준의 성능을 달성하는 것이다.
- Motivation: 기존의 장기 Chain-of-Thought(CoT) 추론 방식이 복잡한 문제에서 중간 오류를 효과적으로 교정하지 못하는 한계를 극복하고자 하였다.
- Contribution: 효율적인 RL 인프라, GRPO-RoC 알고리즘, 그리고 다단계 RL 훈련법을 도입하여 소량의 GPU 자원으로 신속하고 안정적인 agentic 강화학습이 가능하도록 기여하였다.

## Method

rStar2-Agent는 Python 코딩 도구를 활용하여 실행 결과를 반영하며 자율적으로 문제 해결 과정 중간 단계를 탐색하고 검증하는 인지 능력을 강화학습으로 학습한다.  
GRPO-RoC는 환경 잡음을 줄이기 위한 Resample-on-Correct 전략을 도입해 긍정적 경로의 질을 향상시키며 안정적인 정책 업데이트를 가능하게 한다.  
또한 64개의 MI300X GPU를 활용한 고성능 병렬 코드 실행

### 3. [USO: Unified Style and Subject-Driven Generation via Disentangled and

Reward Learning](https://arxiv.org/abs/2508.18966)

제목: USO: Unified Style and Subject-Driven Generation via Disentangled and  
Reward Learning  
arXiv: https://arxiv.org/abs/2508.18966

## Introduction

- Goal: 본 연구는 스타일 주도 및 주체 주도 이미지 생성 과제를 단일 프레임워크 내에서 통합하여 콘텐츠와 스타일의 분리 및 재조합 문제를 해결하는 것이다.
- Motivation: 기존 연구들은 스타일 및 주체 주도 생성을 별개의 문제로 다루어 분리 학습에 한계가 있었으며, 두 과제의 상호 보완적 특성을 이용한 통합 접근이 필요하다.
- Contribution: 본 논문은 크로스 태스크 공동 분리(co-disentanglement) 패러다임과 함께 대규모 삼중항 데이터셋, 진행형 스타일 정렬훈련, 콘텐츠-스타일 분리 학습 및 스타일 보상 학습을 제안하였다.

## Method

본 연구는 주체-스타일 삼중항 데이터셋을 구축하고 스타일 및 콘텐츠 특징 분리를 위한 이중 목표 학습 체계를 도입하여 스타일 정렬과 분리 학습을 수행한다. 이어서 스타일 보상 학습(SRL)을 적용하여 모델 성능을 향상시키며, 다중 조건 입력을 처리하는 USO 통합 모델을 개발하였다. 마지막으로 스타일 일관성과 주

### 4. [AWorld: Orchestrating the Training Recipe for Agentic AI](https://arxiv.org/abs/2508.20404)

제목: AWorld: Orchestrating the Training Recipe for Agentic AI  
arXiv: https://arxiv.org/abs/2508.20404

## Introduction

- Goal: 본 연구의 목표는 복잡한 환경과 상호작용하며 에이전트를 효과적으로 학습시키기 위한 대규모 분산 플랫폼인 AWORLD 프레임워크를 설계 및 구현하는 것이다.
- Motivation: 기존 에이전트 학습에서 경험 생성 과정의 비효율성이 주요 병목으로 작용하여, 특히 복잡한 GAIA 벤치마크에서 학습 속도와 성능 향상에 한계가 존재한다는 점에서 동기가 부여되었다.
- Contribution: AWORLD는 경험 생성 속도를 14.6배 가속하여 대규모 강화학습을 실현하고, 이를 통해 Qwen3-32B 기반 에이전트를 효과적으로 훈련시켜 기존 모델 대비 크게 향상된 GAIA 벤치마크 성능을 달성하였다.

## Method

AWORLD는 에이전트 구축, 통신 프로토콜, 분산 실행, 학습 조율 네 가지 핵심 요소로 구성된다. 에이전트는 다중 도구와 상호작용하며 메시지 기반 커뮤니케이션을 통해 동적 협업을 지원하고, 쿠버네티스를 활용한 분

### 5. [TCIA: A Task-Centric Instruction Augmentation Method for Instruction

Finetuning](https://arxiv.org/abs/2508.20374)

제목: TCIA: A Task-Centric Instruction Augmentation Method for Instruction  
Finetuning  
arXiv: https://arxiv.org/abs/2508.20374

## Introduction

- Goal: 본 연구는 대형 언어 모델의 지시어 미세조정을 위해 다양성과 과제 적합성을 동시에 유지하는 과제 중심 지시어 증강 방법을 제안하는 것을 목표로 한다.
- Motivation: 기존 자동 지시어 생성 방식은 지시어 다양성은 확보하나 현실 적용에 중요한 과제 관련성(task relevance)을 충분히 고려하지 못해 실무적 분기점(task drift) 문제가 존재한다.
- Contribution: TCIA라는 체계적 프레임워크를 도입하여, 지시어를 분해 및 데이터베이스화하고, 유사과제 제약 조건을 기반으로 한 BFS 탐색을 통해 과제 적합성과 다양성을 보존한 지시어 확장 방법을 제안하였다.

## Method

지시어를 기초 질의와 명확한 제약 조건 집합으로 분해한 후, 대규모 지시어 데이터베이스에서 유사한 과제 및 제약을 검색하여 BFS 알고리즘으로 다양한 제약 상태를 생성한다. 생성된 상태는 다시 자연어 지시어로 복원되고, LLM 기반 검증 및 다차원 샘플링을

### 6. [Mixture of Contexts for Long Video Generation](https://arxiv.org/abs/2508.21058)

제목: Mixture of Contexts for Long Video Generation  
arXiv: https://arxiv.org/abs/2508.21058

## Introduction

- Goal: 본 논문은 장시간 길이의 비디오 생성에서 효율적이고 일관된 장기 문맥 기억을 가능하게 하는 학습 가능한 희소 주의집중 라우팅 모듈인 Mixture of Contexts (MoC)를 제안하는 것을 목표로 한다.
- Motivation: 기존의 확산 트랜스포머는 자기주의의 이차 계산 비용 문제로 인해 긴 시퀀스 생성에 적용하기 어려우며, 고정된 희소 패턴은 중요한 과거 이벤트 선택에 유연하지 못해 장기 의존성 유지가 제한된다.
- Contribution: MoC는 비디오의 자연 경계에 맞춘 내용 정렬 청크를 이용해 각 쿼리가 관련성 높은 일부 청크만 선택하도록 학습하며, 인과 마스킹으로 루프 폐쇄를 방지하고 분산 연산량을 7배 이상 줄이는 동시에 분당 단위 장기 기억 능력을 달성한다.

## Method

장면, 샷, 캡션 토큰으로 나누어진 다중 모달 시퀀스를 내용 기반 청크로 분할하고, 각 쿼리가 상위 k개의 관련 청크와 필수 앵커

### 7. [MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World

Tasks via MCP Servers](https://arxiv.org/abs/2508.20453)

제목: MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World  
Tasks via MCP Servers  
arXiv: https://arxiv.org/abs/2508.20453

## Introduction

- Goal: 본 연구는 대규모 언어 모델(LLM) 에이전트를 현실적이고 복잡한 다중 단계 도구 사용 작업에서 평가하기 위한 MCP-Bench 벤치마크를 제안한다.
- Motivation: 기존 도구 사용 벤치마크들이 도구 간 자연스러운 의존성 및 장기 계획 능력, 불명확한 지시문 하 도구 선택 능력 등을 충분히 반영하지 못하는 한계를 극복하기 위함이다.
- Contribution: MCP 서버 28개, 250개 도구를 연결하여 실제 생태계를 기반으로 한 대규모 복합 작업들을 자동 생성하고, 엄격한 평가 프레임워크를 통해 LLM 에이전트의 도구 이해, 계획, 종합 수행 능력을 종합적으로 측정한다.

## Method

MCP-Bench는 MCP 프로토콜을 활용하여 다양한 도메인의 MCP 서버와 250개 도구를 연동하며, LLM 기반 자동 작업 합성 파이프라인을 통해 현실적이고 다중 서버 연계가 가능한 작업을 생성한다.  
평가에는 도구 이름 유효성,

### 8. [OnGoal: Tracking and Visualizing Conversational Goals in Multi-Turn

Dialogue with Large Language Models](https://arxiv.org/abs/2508.21061)

제목: OnGoal: Tracking and Visualizing Conversational Goals in Multi-Turn  
Dialogue with Large Language Models  
arXiv: https://arxiv.org/abs/2508.21061

## Introduction

- Goal: 다중 대화에서 대화 목표를 추적하고 시각화하여 사용자들이 대화 목표 진행 상황을 효율적으로 평가 및 검토하도록 지원하는 인터페이스를 제안하는 것이다.
- Motivation: 장기적이고 복잡한 LLM 다중 턴 대화에서 사용자들이 목표가 어떻게 달성되고 있는지 파악하기 어렵고, 이로 인해 반복적인 프롬프트나 대화 재시작 등의 문제를 겪기 때문이다.
- Contribution: OnGoal이라는 목표 추적 파이프라인과 이를 시각화하는 채팅 UI를 개발하고 20인 참가자 연구를 통해 목표 피드백이 대화 참여도와 회복력을 높이며, 향후 인터페이스 디자인에 관한 시사점을 도출하였다.

## Method

- OnGoal은 LLM을 활용한 3단계 목표 파이프라인(추론, 병합, 평가)을 통해 사용자의 대화 목표를 실시간으로 추출, 통합, 평가한다.
- 이 파이프라인 결과는 인라인 목적 글리프, 목표 진행 타임라인, 상세 목표 검토 뷰 등 다양한 시각화 도

### 9. [CogVLA: Cognition-Aligned Vision-Language-Action Model via

Instruction-Driven Routing & Sparsification](https://arxiv.org/abs/2508.21046)

제목: CogVLA: Cognition-Aligned Vision-Language-Action Model via  
Instruction-Driven Routing & Sparsification  
arXiv: https://arxiv.org/abs/2508.21046

## Introduction

- Goal: 본 연구는 Vision-Language-Action(VLA) 모델의 높은 계산 복잡도를 줄이고, 인지적 일치 기반의 통합적 멀티모달 처리 방식을 제안하는 데 있다.
- Motivation: 기존 VLA 모델은 시각-언어-행동 간 의미적 연계성을 간과하여 효율성 및 일관성 측면의 한계를 보였기 때문이다.
- Contribution: 인간 인지 시스템에서 영감을 받은 3단계 진화적 구조와 명령어 기반 라우팅 및 희소화 기법(EFA-Routing, LFP-Routing, CAtten)을 통합하여 성능과 효율을 동시에 개선하였다.

## Method

CogVLA는 1) 명령어 정보를 시각 인코더에 주입해 시각 토큰을 선택적으로 압축하는 EFA-Routing, 2) 언어 모델 내에서 명령어와 관련이 없는 시각 토큰을 제거하는 LFP-Routing, 3) 압축된 멀티모달 정보를 논리적 일관성과 행동 연속성 유지가 가능한 V-L-A 결합 주의 메

### 10. [Dress&Dance: Dress up and Dance as You Like It - Technical Preview](https://arxiv.org/abs/2508.21070)

제목: Dress&Dance: Dress up and Dance as You Like It - Technical Preview  
arXiv: https://arxiv.org/abs/2508.21070

## Introduction

- Goal: Dress&Dance는 단일 사용자 이미지, 원하는 의상, 참조 동영상을 입력받아 고해상도 5초 가량의 24FPS 가상 착용 영상을 생성하는 비디오 확산 프레임워크이다.
- Motivation: 기존의 가상 착용 기술은 주로 정적인 2D 이미지 생성에 한정되어 사용자가 의상의 착용감과 움직임을 충분히 경험하지 못하는 한계가 존재한다.
- Contribution: 본 연구는 다중 모달 입력을 통합하는 주목(attention) 기반 조건 네트워크 CondNet을 도입하고, 단계적 해상도 향상 및 합성 트리플렛 데이터로 효율적인 학습 전략을 제안하여 고품질의 동영상 가상 착용을 실현하였다.

## Method

Dress&Dance는 사용자 이미지, 의상 이미지, 모션 참조 영상, 선택적 텍스트를 토큰 시퀀스로 인코딩하여 통합 확산 백본 모델에 입력한다. CondNet은 이질적 입력을 동질적인 주목 시퀀스로 변환해 모델에 통합 처리함으로써 의상 등록과 모션 일
