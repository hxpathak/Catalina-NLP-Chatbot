""" Name: Hemal Pathak
    CS 6320.001 Project 1 - Chatbot
    3/4/2024
"""

# Importing necessary libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
import spacy
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Storing file of pickled knowledge base
kb_pickled = "knowledge_base"

# Reading in pickled knowledge base
with open(kb_pickled, 'rb') as handle:
    knowledge_base = pickle.load(handle)


# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Initializing list of English and Spanish stopwords
stopwords_spanish = stopwords.words('spanish')
stopwords_eng = stopwords.words('english')
stopwords_eng.append("ex")

# Create user model dictionary
personal_info = {}

# User input string
user_input = ''

# Lists of possible user responses
end_convo = ['bye', 'goodbye', 'end', 'talk later', 'exit', 'adios', 'later', 'done']
sa_countries = ['colombia', 'argentina', 'brazil', 'uruguay', 'paraguay', 'chile']
yes_options = ['yes', 'yeah', 'yup', 'of course', 'yes please', 'sure', 'ok']
no_options = ['no', 'nah', 'nope', 'not at all', 'negative']

# Flags to check flwo of conversation
greeting_flag = 1
country_flag = 1
experience_flag = 1
activities_flag = 1
dislike_flag = 1
user_query_flag = 0
suggestion_flag = 1
iteration_num = 0
user_input_error_flag = 0

# Check if user's input contains any end_convo words
def check_end_convo(user_input):
    return any(phrase in user_input.lower() for phrase in end_convo)

def print_greeting():
    """
        Function to randomly print a greeting
    """
    greetings = ["CatalinaCB: Hola! I'm Catalina, your South America travel guide chatbot. (To end the conversation, please type 'bye' or 'end')\nCatalinaCB: What's your name?", "CatalinaCB: Hi! My name is Catalina and I'm a travel guide for South America. (To end the conversation, please type 'bye' or 'end')\nCatalinaCB: What's your name?",
                 "CatalinaCB: Hey there! I'm Catalina, a chatbot for visiting South America. (To end the conversation, please type 'bye' or 'end')\nCatalinaCB: What's your name?", "CatalinaCB: Hello fellow explorer! I'm Catalina, a travel guide chatbot for South America. (To end the conversation, please type 'bye' or 'end')\nCatalinaCB: What's you name?"]
    print(greetings[random.randint(0, 3)])

def print_greeting_again():
    """
        Function to randomly print a greeting
    """
    greetings = ["CatalinaCB: Hi again! I didn't quite get your name. Can you tell me your name again?", "CatalinaCB: I'm excited to talk to you today. However, I didn't get your name. Can you tell me your name again?", 
                 "CatalinaCB: Hi there again! I apologize, but I didn't get your name. Can you tell me your name again?"]
    print(greetings[random.randint(0, 2)])


def check_countries(user_countries):
    """
        Check if user's country request is supported
        Arguments: list of countries provided by user
        Output:
            supported - list of countries provided by user that are supported
            not_supported - list of countries provided by user that are not supported
    """
    # Initialize lists
    not_supported = []
    supported = []

    # Iterate through countries provided by user
    for country in user_countries:
        if country not in sa_countries:
            not_supported.append(country)
        else:
            supported.append(country)
    return supported, not_supported



def preprocess(text):
    """
        Preprocess function that lemmatizes, removes stopwords in English and Spanish, and removes numbers
        Arguments: text string to be preprocessed
        Output: 
            ' '.join(cleaned_tokens) - string of cleaned tokens after preprocessing
    """
    lemmatize = nlp(text.lower())
    lemmatized_tokens = [token.lemma_ for token in lemmatize]
    cleaned_tokens = [token for token in lemmatized_tokens if token not in stopwords_spanish and token not in stopwords_eng and token.isalpha()]
    return ' '.join(cleaned_tokens)


def get_synonyms(term):
    """
        Get list of synonyms, hypernyms, and hyponyms of a term 
        Arguments: term
        Output: 
            synonyms - list of all related words
    """
    synonyms = set()
    for synset in wordnet.synsets(term):
        # Clean and add lemma names
        for l in synset.lemmas():
            synonyms.add(clean_lemma_name(l.name()))  

        for hypernym in synset.hypernyms():
            # Clean and add hypernym names
            for l in hypernym.lemmas():
                synonyms.add(clean_lemma_name(l.name()))  

        for hyponym in synset.hyponyms():
            # Clean and add hyponym names
            for l in hyponym.lemmas():
                synonyms.add(clean_lemma_name(l.name())) 
    return synonyms
 

def clean_lemma_name(lemma_name):
    """
        Clean string to not contain "_" or "-"
        Arguments: string of word
        Output: 
            name - cleaned string
    """
    # Splits the string and takes the first part
    name = lemma_name.split('.')[0].lower()  

    # Replacing "_" or "-"
    if('-' in name or '_' in name):
        name = name.replace('-', ' ')
        name = name.replace('_', ' ')
    return name


def find_relevant_response(user_request):
    """
        Using cosine similarity and tfidf vectorization, find a sentence that most closely matches the user's request
        Arguments: string of text
        Output: 
            relevant_sentence - sentence string
            0 - when user request contains no relevant information
    """
    # Create string to add cities to user_preference based on country
    cities = ''
    if("argentina" in personal_info['country']):
        cities += "bariloche salta cafayate cordoba fitz beagle"
    elif("brazil" in personal_info['country']):
        cities += "iguazu"
    elif("chile" in personal_info['country']):
        cities += "natale pedro fitz"
    elif("paraguay" in personal_info['country']):
        cities += "cuidad"
    user_preference = ' '.join(personal_info['country']) + ' ' + cities

    # Initialize tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Create list to store vectorized output
    user_vector = []

    # If user is querying, then find a sentence match
    if(user_query_flag):
        # Preprocess request
        preprocessed_user = preprocess(user_request)

        # If resulting text is empty, return 0
        if(len(preprocessed_user) == 0):
            return 0

        # Get synonyms of key terms
        trek_syn = get_synonyms('trek')
        wine_syn = get_synonyms('wine')
        car_syn = get_synonyms('car')

        # Check if any related words in user request, if so add corresponding term to user_preference
        preprocess_user_tokens = word_tokenize(preprocessed_user)
        if('hike' in preprocessed_user or 'walk'  in preprocessed_user or 'backpack' in preprocessed_user or any(word in preprocess_user_tokens for word in trek_syn)):
            user_preference += " trek"
        if('food' in preprocessed_user or 'cuisine' in preprocessed_user or any(word in preprocess_user_tokens for word in wine_syn)):
            user_preference += " wine"
        if('roadtrip' in preprocessed_user or 'transportation'  in preprocessed_user or any(word in preprocess_user_tokens for word in car_syn)):
            user_preference += " car"
        
        # tfidf vectorize user request
        user_vector = tfidf_vectorizer.fit_transform([preprocessed_user])
    else:
        # Giving user a suggestion
        user_preference += ' ' + ' '.join(personal_info['likes'])

        # Preprocess
        preprocessed_user = preprocess(user_preference)

        # tfidf vectorize user preferences
        user_vector = tfidf_vectorizer.fit_transform([preprocessed_user])
    
    # Create string to store sentence
    relevant_sentence = ''
    # Comparison for cosine similarity value
    tmp_max = -1

    # Iterate through knowledge base
    for keyword, sentences in knowledge_base.items():
        if(keyword in user_preference):
            # Preprocess
            preprocessed = [preprocess(each_sentence) for each_sentence in sentences]

            # tfidf vectorize
            text_vector = tfidf_vectorizer.transform(preprocessed)

            # calculate cosine similarity
            cos_sim = cosine_similarity(user_vector, text_vector)

            # Find highest value
            highest_similarity = max(cos_sim[0])

            # Store index for sentence with highest cosine similarity value
            highest_similarity_index = cos_sim[0].tolist().index(highest_similarity)

            # Find max cosine similarity value and verify sentence doesn't include user dislikes
            if(highest_similarity > tmp_max and not any(word in sentences[highest_similarity_index] for word in personal_info['dislikes'])):
                tmp_max = highest_similarity
                relevant_sentence = sentences[highest_similarity_index]

    # If cosine similarity value lower than 0.5 threshold, then print clarifying message
    if(highest_similarity < 0.5):
        print("CatalinaCB: Sorry, I don't have enough information for your request, but this is the closest response I have:")
    return relevant_sentence




# Chatbot dialogue
while(not check_end_convo(user_input)):
    # Get user's name
    if greeting_flag:
        print_greeting()
        user_input = input()
        doc = nlp(user_input)
        name = ''

        # Find proper noun
        for token in doc:
            if token.pos_ == 'PROPN':
                name = str(token)
                personal_info['name'] = name
                greeting_flag = 0
                break
        
        # Continue asking user for their name
        if not check_end_convo(user_input):
            while(greeting_flag):
                print_greeting_again()
                get_name = input()
                doc = nlp(get_name)

                # Find a proper noun
                for token in doc:
                    if token.pos_ == 'PROPN':
                        user_input = get_name
                        name = str(token)
                        personal_info['name'] = name
                        greeting_flag = 0
                        break
        else:
            break
        # Ask user for countries
        print("CatalinaCB: Hi " + personal_info['name'] + 
          "! If you're looking to travel to South America, you've come to the right place! What country(s) are you looking to visit in South America?")
        user_input = input().lower()

    # Get user's country preferences
    if(not check_end_convo(user_input)):
        while(country_flag and not check_end_convo(user_input)):
            # Find all entities labeled as 'GPE'
            doc = nlp(user_input.lower())
            cur_countries = []
            for entity in doc.ents:
                if entity.label_ == 'GPE' and entity.text not in cur_countries:
                    cur_countries.append(str(entity.text))
            
            # Spacy model doesn't pick up on these countries. If listed by user, then add to list
            if 'uruguay' in user_input.lower() and 'uruguay' not in cur_countries:
                cur_countries.append('uruguay')
            if 'chile' in user_input.lower() and 'chile' not in cur_countries:
                cur_countries.append('chile')
            if 'paraguay' in user_input.lower() and 'paraguay' not in cur_countries:
                cur_countries.append('paraguay')
            if 'argentina' in user_input.lower() and 'argentina' not in cur_countries:
                cur_countries.append('argentina')

            # Verify if all countries provided are supported
            supported_countries, not_supported_countries = check_countries(cur_countries)

            # Print message to user if country(s) not supported
            if(len(not_supported_countries) > 0):
                str_yes = ', '.join(sa_countries)
                if(len(not_supported_countries) == 1):
                    str_no = ''.join(not_supported_countries)
                else:
                    str_no = ', '.join(not_supported_countries)
                print("CatalinaCB: Great choice! Unfortunately, at this time, I can't provide more information on: " + str_no + 
                    ". I'm only familiar with these countries: " + str_yes + ". Would you like to continue discussing about any of these countries?")
            
            # Print message if no countries provided
            elif(len(not_supported_countries) == 0 and len(supported_countries) == 0):
                str_yes = ', '.join(sa_countries)
                print("CatalinaCB: Hmm... I didn't quite get the countries you want to visit. Here's a list of countries I'm familiar with: " + str_yes + 
                    ". Which of these countries would you like to talk about?")
            
            # Continue conversation if all countries are supported, ask about experience level
            elif(len(not_supported_countries) == 0 and len(supported_countries) > 0):
                if(len(supported_countries) == 1):
                    str_yes = ''.join(supported_countries)
                    print("CatalinaCB: Perfect! " + str_yes + " is a beautiful place to visit! What would you rate your travel experience level: beginner, intermediate, advanced?")
                else:
                    str_yes = ', '.join(supported_countries)
                    print("CatalinaCB: Perfect! " + str_yes + " are beautiful places to visit! What would you rate your travel experience level: beginner, intermediate, advanced?")
                personal_info["country"] = cur_countries
                country_flag = 0
            
            user_input = input().lower()
    else:
        break
    
    # Get user travel experience level
    if(not check_end_convo(user_input)):
        while(experience_flag and not check_end_convo(user_input)):
            # Create count variable
            count = 0

            # Find level in user input
            if('beginner' in user_input):
                personal_info["experience"] = 'beginner'
                count += 1
                experience_flag = 0
            if('intermediate' in user_input):
                personal_info["experience"] = 'intermediate'
                count += 1
                experience_flag = 0
            if ('advance' in user_input):
                personal_info["experience"] = 'advance'
                count += 1
                experience_flag = 0
            
            # Check if more than 1 level provided
            if(count > 1):
                experience_flag = 1
            
            # Print message when valid experience level is identified
            if(not experience_flag):
                print("CatalinaCB: Awesome! Glad you're ready to explore " + personal_info["name"] + "! What things do you like when traveling (activities, places, etc.)?")
            else:
                # Print message when invalid experience level provided and continue requesting user input
                print("CatalinaCB: Hmm.. can you share your travel experience level again? Please enter 1 level")
                user_input = input().lower()
    else:
        break

    # Create list of synonyms of 'dislike'
    dislike_list = get_synonyms('dislike')
    dislike_list.add('not')

    # Get user likes
    while(activities_flag and not check_end_convo(user_input)):
        user_input = input().lower()
        if(not check_end_convo(user_input)):
            # Check if user provided a dislike
            if(any(word in user_input.lower() for word in dislike_list)):
                print("CatalinaCB: I'll ask you about your dislikes next, so save that thought for later! I'd love to know what you like when traveling (activities, places, etc.)")
            else:
                # Find all nouns and verbs with spacy POS tagging
                doc = nlp(user_input)
                activities = [token.text for token in doc if token.pos_ == "VERB" and token.text != 'like']
                likes = [token.text for token in doc if token.pos_ == "NOUN"]

                # Print message when no nouns or verbs provided
                if(len(activities) == 0 and len(likes) == 0):
                    print("CatalinaCB: Feel free to share what you enjoy while traveling!")
                else:
                    # Print message when one or more experience nouns or verbs provided, store preferences in user model
                    personal_info['likes'] = likes + activities
                    if(len(activities) == 0):
                        print("CatalinaCB: " + likes[0] + " sounds like fun! Now, what things do you dislike when traveling?")
                    else:
                        print("CatalinaCB: " + activities[0] + " sounds like fun! Now, what things do you dislike when traveling?")
                    activities_flag = 0
        else:
            break
    
    # Get user dislikes
    while(dislike_flag and not check_end_convo(user_input)):
        user_input = input().lower()
        if(not check_end_convo(user_input)):
            # Find all nouns and verbs with spacy POS tagging
            doc = nlp(user_input)
            dislikes = [token.text for token in doc if token.pos_ == "VERB" or token.pos_ == "NOUN"]

            # Print message when no nouns or verbs provided
            if(len(dislikes) == 0):
                print("CatalinaCB: Feel free to share what you don't enjoy doing while traveling!")
            else:
                # Print message when one or more experience nouns or verbs provided, store preferences in user model
                personal_info['dislikes'] = dislikes
                dislike_flag = 0
                print("CatalinaCB: Thank you for sharing! Based on your responses, I'll give you a suggestion:")
        else:
            break
    
    # Provide a suggestion or response to user query
    if(not check_end_convo(user_input)):
        if(user_query_flag):
            # Find 'GPE' entities using NER
            doc = nlp(user_input)

            # Add country if indicated by user, 
            cur_countries = personal_info['country']
            for entity in doc.ents:
                if entity.label_ == 'GPE' and entity.text not in cur_countries and entity.text in sa_countries:
                    cur_countries.append(str(entity.text))
            if 'uruguay' in user_input and 'uruguay' not in cur_countries:
                cur_countries.append('uruguay')
            if 'chile' in user_input and 'chile' not in cur_countries:
                cur_countries.append('chile')
            if 'paraguay' in user_input:
                cur_countries.append('paraguay')

            # Perform cosine similarity of user query and knowledge base
            chatbot_response = find_relevant_response(user_input)

            # Print message if user request doesn't have enough information
            if(chatbot_response == 0):
                print("CatalinaCB: Sorry, I didn't get your request. Try including names of places, things, or activities in your request so I can best help you!")
                user_input = input()
                user_input_error_flag = 1
            else:
                # Print result of cosine similarity (best matching sentence)
                print("CatalinaCB: " + chatbot_response)
                user_input_error_flag = 0
                iteration_num += 1
        else:
            # Provide user a suggestion
            print("CatalinaCB: " + find_relevant_response(''))
            print("CatalinaCB: Would you like to continue dicussing this?")
            user_input = input()

        # Follow up to suggestion
        if(not check_end_convo(user_input)):
            while(suggestion_flag):
                # Get user response: yes or no
                if(any(response in user_input.lower() for response in yes_options) or any(response in user_input.lower() for response in no_options)):
                    print('CatalinaCB: Ok! What can I help you with ' + personal_info['name'] + '?')
                    user_query_flag = 1
                    suggestion_flag = 0
                    user_input = input()
                else:
                    # Confirm if user wants to continue dicussing suggestion
                    if(not check_end_convo(user_input)):
                        print('CatalinaCB: Would you like to continue dicussing the previous suggestion?')
                        user_input = input()
                    else:
                        break
        else:
            break
        
        # Allow user queries until they end conversation
        if(iteration_num > 0 and not user_input_error_flag):
            prompts = ["CatalinaCB: What else would you like to know?", "CatalinaCB: Feel free to ask me anything else!", 
                    "CatalinaCB: What other information can I provide for you?", "CatalinaCB: Is there anything else you're curious about?", 
                    "CatalinaCB: What else can I help you with?"]
            print(prompts[random.randint(0, 4)])
            user_input = input().lower()
        

    else:
        break
    
    
        
# Print end message after conversation
print('CatalinaCB: Thanks for chatting with me! Hope to see you again:)')


# Create user model fiel
user_model = "user_model.txt"

# Write personal_info dictionary to file
with open(user_model, 'w') as file:
    file.write(str(personal_info))

