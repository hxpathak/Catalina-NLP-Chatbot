# Catalina-NLP-Chatbot
Hi there! I'm Catalina - a chatbot for traveling in South America! ✈️☀️💃

# Overview
A conversational rules-based chatbot created on the topic of traveling in South America, using various Natural Language Processing techniques such as TF-IDF vectorization and cosine similarity. The chatbot has 2 parts: the knowledge base and the conversation flow.

# Knowledge Base
The knowledge base is what the chatbot uses to provide answers to user's queries. To create the knowledge base, a web scraper was built to extract content and links from related websites. One starter url was used, and urls on this page were added to a list. All urls in the list were scraped to extract information and other related links.

The content from websites were cleaned using NLP techniques like NER, tokenization, and regex to retain information related to the topic and to disregard unrelated content.

After the content was cleaned, the top 40 terms were identified using TF-IDF frequencies. The top 15 most important words were selected, and a dictionary was created to map each sentence with the extracted text to a key word. This dictionary was the knowledge base.

# Chatbot
The second part is a the chatbot, which provides the conversation flow. The chatbot asks the user various personal questions, such as their name, likes, dislikes, travel experience, and stores this information in a user model.

Based on the user's preferences, the chatbot provides a travel recommendation by going through the knowledge base and identifying a sentence which includes a keyword that matches the user's preferences.

Similarly, when a user provides a query, the chatbot cleans and extracts key words from the query and tries to find a matching sentence within the database based on the query.
