# FinanceRAG System for Mixed Data Types: Exploring Structured and Unstructured Data Processing
Project work for the course of NLP, MSc at University of Bologna

![GitHub](https://img.shields.io/badge/status-in_development-blue)

## 🎯 Project Overview

This project explores the capabilities and limitations of Retrieval-Augmented Generation (RAG) systems when handling mixed datasets containing both textual and structured (tabular) data. While RAG systems excel at processing unstructured text, they often struggle with numerical and tabular data that require more specific handling and precise context.

## 🔍 Key Objectives

- Implement and evaluate standard RAG pipelines for document retrieval and ranking
- Develop advanced methodologies for handling both structured and unstructured data
- Leverage generative LLMs for creating insights and interpreting tabular data
- Compare different approaches on mixed datasets
- Evaluate retrieval effectiveness using available labels

## 📊 Dataset

The project utilizes the [FinanceRAG dataset](https://huggingface.co/datasets/Linq-AI-Research/FinanceRAG), which contains various documents in both purely textual and mixed formats (text + tabular data).

## 🛠️ Technical Approach

### Phase 1: Standard Pipeline
- Implementation of baseline RAG systems
- Standard document retrieval and ranking approaches
- Evaluation of basic performance metrics

### Phase 2: Advanced Methodologies
- Integration of LLM-powered insight generation
- Enhanced tabular data interpretation
- Implementation inspired by [LangChain's multi-vector retriever](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- Comparative analysis of performance improvements

## 📈 Features

- Dual-mode processing for both structured and unstructured data
- LLM-enhanced insight generation
- Advanced embedding techniques for mixed data types
- Comprehensive evaluation framework
- Performance comparison metrics

## 🔬 Evaluation

The project includes a robust evaluation framework leveraging:
- Available labeled data for retrieval accuracy assessment
- Comparative metrics between standard and advanced approaches
- Performance analysis on mixed data types

## 📜 License
MIT License <br>
Copyright (c) 2025 Alessio Conti

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 📞 Contact
[Alice Turrini](alice.turrini@studio.unibo.it) <br>
[Alessio Conti](alessio.conti3@studio.unibo.it)
