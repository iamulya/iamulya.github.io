---
layout: postwithads
title: End-to-End Serverless RAG Application on GCP
date: 2024-08-08 10:00:00 +0100
categories: [Gen AI, RAG Applications, Serverless]
tags: [Google Cloud, RAG, Serverless, Cloud Workflows]
---

# End-to-End Serverless RAG Application on GCP

Retrieval-Augmented Generation (RAG) is an approach in LLM-based Applications which enables an LLM like Gemini to answer queries regarding topics it wasn’t even trained with. This is done by **augmenting** the context of a query/prompt with the data necessary to answer the query, **retrieved** from an external source like a Database, so that the LLM can **generate** an answer. The data can be in the form of text, audio, images and even videos - basically anything that you can create embeddings for.

In this article, we’ll see how you can create your RAG Application using only serverless technologies on Google Cloud. This approach has many advantages, including but not limited to, performing cheap PoCs, high availability and scalability.

## Anatomy of RAG

RAG applications need a few steps to be performed before queries can be answered. These are:

1. **Data Chunking** - Parsing and splitting the unstructured data into meaningful chunks. This is extremely necessary because of multiple reasons, including but not limited to, LLM model’s token limits and different chunking/splitting approaches based on the type of data. The quality of your chunks is quite integral to the quality of answers an LLM can generate. A smart chunking approach yields a smart RAG application.
2. **Embed and Store** - Embed the chunks you created earlier using an Embedding model like VertexAIEmbeddings and store them in a vector storage solution.

Other than these steps, you will also need to store the status of chunking/embedding of your data, so that you can let the user know when the data is ready to be queried.

Now that you have stored the embeddings, you can start with the query phase which includes:

1. **Retrieval** - retrieve the chunks which are relevant to the answer of your query. There are many retrieval strategies (similarity search, hybrid, router, automerge, recursive etc.) out there depending on the structure and complexity of your data. This step will basically decide how well your RAG application will perform.
2. **Generation** - Once you have the relevant chunks for your query, you add them to your query/prompt as a supplement, providing context to the LLM which then accordingly generates a response.

## Serverless RAG

Now let’s create a serverless RAG application, where you can upload documents and once  they are embedded/indexed, you can have a Q&A session with it. We’ll keep it simple for this article’s purposes but it should be very easy to adapt the code for more complex scenarios.

Finally, It’s time to introduce the serverless products that will help us perform all the necessary tasks.

- **BigQuery** - Serverless Data Warehouse with support for saving embedding/vectors.
- **Cloud Storage** - Serverless storage for your unstructured data. This is where the upload of the document will happen. Each user will have its own folder inside a designated bucket - we call it a collection.
- **Document AI** - Document processing service used to parse the documents.
- **Firestore** - Serverless document database for saving the document metadata.
- **Cloud Functions** - Serverless Compute for running the necessary tasks. We’ll write the following functions:
    - *Pre-signed URL*: Generates a pre-signed URL, given a collection and the filename to be uploaded.
    - *Chunker*: Parses the uploaded document and split it into chunks. Document AI will be used to parse the document and will be split at the paragraph level. All the document metadata will be saved in Firestore.
    - *Indexer*: Reads the data out of Firestore for a specific document, embeds the data in batches and stores the result into BigQuery Vector Store. It also accordingly updates the index status of all the chunks who embeddings have been successfully saved. This status can be polled by a frontend in order to know when a document is completely indexed and the query process can start.
    - *Query*: Uses the “similarity search” retriever to retrieve the data similar to the query. That data is then sent along with the query to Gemini Pro on Vertex AI to generate an answer.
- **Cloud Workflows** - Serverless orchestration tool where we put most of it together. The workflow gets triggered automatically when a file gets uploaded into a designated bucket using the Pre-signed URL Cloud Function. After this, the workflow calls the *Chunker* function, before *Indexer* kicks in.

The use of Cloud Workflows also makes it easy to do error-handling, exponential-backoff retry in case of Quota issues, without writing any extra code! 

You can find the code [here](https://github.com/iamulya/gcp-serverless-rag). It uses Langchain as the LLM-Framework but you can easily adjust it to use another Framework like Llamaindex.

## Next Steps

Congratulations, you have built a serverless RAG application, albeit a very basic one. You can now expand this code to do more advanced stuff:

1. Use different chunking techniques based on the type of data you input. For e.g. if you want to build a Q&A system for your source code, libraries like Llamaindex and Langchain have specific splitters for formats like Markdown, HTML, Programming languages like Javascript, Python etc. for which you can add support in the *Chunker* function.
2. If the answers you are getting are not satisfactory, you either need to adjust your chunking strategy or retrieval strategy (in some cases both!). Llamaindex has support for a lot of advanced retrieval strategies which work well for complex structures. Add support for more retrieval strategies in the *Indexer*.
3. Once you are out of the PoC mode and ready to increase the scope of your RAG, make sure your RAG application has enough Quota for the Prediction and Embedding calls. While the architecture is highly scalable, if you don’t have enough Quota, the Indexer will get stuck and the workflow will have to do multiple retry calls. Under high load, the Indexing process will be too slow if enough Quotas aren’t available.
4. Before going to production, make sure to secure your Vertex AI API calls with VPC Service Controls. Also make sure that your service accounts have the least permissions needed to perform the tasks.