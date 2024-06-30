""" Name: Hemal Pathak
    CS 6320.001 Project 1 - Web Crawler
    3/4/2024
"""

# Importing necessary libraries
import re
import math
from urllib import request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet 
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import unidecode
import spacy
import pickle

# Loading spacy small model
nlp = spacy.load('en_core_web_sm')

# Initializing list of English and Spanish stopwords
stopwords_spanish = stopwords.words('spanish')
stopwords_eng = stopwords.words('english')
stopwords_eng.append("ex")

# List of social media names for scraper to disregard
social_media = ['instagram', 'facebook', 'twitter', 'linkedin', 'reddit', 'youtube', 'tumblr', 'pinterest', 'flipboard', 'fave', 'flickr', 'tiktok', 'airbnb']

# List of other countries from travel blog for scraper to disregard
other_places = ['africa', 'southeast-asia', 'north-america', 'middle-east', 'europe', 'central-america', 'asia', 'australia', 'maui']

# List of countries/places relevant to traveling in South America
sa_places = ['argentina', 'brazil', 'uruguay', 'paraguay', 'colombia', 'chile', 'iguazu', 'patagonia']

# List of words in sites that are irrelevant to traveling in South America
irrelevant_sites = ['book', 'rental', 'car', 'budget.com', 'home', 'Hotel', 'hotel', 'resort', 'hostel', 'getyourguide', 'tripadvisor', 'viator', 'kimmie', 'avant', 'argentina4u', 'ibtimes', 'skyscanner', 'denomades', 'tourradar', 'vfsglobal', 'hertz', 'laaldeadelaselva', 'stylish', 'polo', 'globeguide', 'wildlifediaries', 'prf', 'badbrother', 'welcomeargentina', 'unesco', 'solsalute', 'nztraveltips']

# Create list to hold all urls
url_list = []

# Get each country links from starter link using BeautifulSoup library
starter_url = 'https://www.adventuresnsunsets.com/south-america-places-to-visit/'
starter_html = request.urlopen(starter_url).read().decode('utf8')
starter_soup = BeautifulSoup(starter_html, 'html.parser')

# Create counter variable to count number of urls
counter = 0

# Find all urls on each site that are specific to each country
for link in starter_soup.find_all('a'):
    link_str = str(link.get('href'))
    media_check = any(media in link_str for media in social_media)

    # Verify and add url to list if url doesn't contain social media, starts with http, and contains "south-america"
    if link_str != "None" and link_str.startswith('http') and not media_check and 'south-america/' in link_str:
        if link_str not in url_list:
            url_list.append(link_str)
            counter += 1


# Crawl among country links
counter = 1
country_limit = len(url_list)

# Define limit for internal domain and external domain links
internal_limit = 4
external_limit = 4

# Iterate through all scraped urls in url_list to find urls for sites that contain content
for url in url_list:
    # If 20 relevant urls found, then stop scraping
    if (len(url_list) - country_limit) == 20:
        break
    else:
        # HTML parse website with BeautifulSoup
        html = request.urlopen(url).read().decode('utf8')
        soup = BeautifulSoup(html, 'html.parser')

        # Get url domain name
        url_domain = urlparse(url).netloc

        # Scrape website for relevant links and set variables to count number of internal and external domain urls
        internal_links = 0
        external_links = 0
        for link in soup.find_all('a'):
            # If 20 relevant urls found, then stop scraping
            if((len(url_list) - country_limit) == 20):
                break
            else:
                # Check if url doesn't contain social media or places other than the supported South America places defined in sa_places
                link_str = str(link.get('href'))
                media_check = any(media in link_str for media in social_media)
                other_place_check = any(place in link_str for place in other_places)
                sa_check = any(loc in link_str for loc in sa_places)
                if link_str != "None" and link_str.startswith('http') and link_str not in url_list and not media_check:
                    # Check if all country websites have been scraped 
                    if counter <= country_limit:
                        # Check if url contains any relevant words and if internal_links limit has not been reached
                        if sa_check  and not other_place_check and 'festival' not in link_str and ('itinerary' in link_str or 'things' in link_str or 'visit' in link_str or 'guide' in link_str) and internal_links < internal_limit:
                            url_list.append(link_str)
                            internal_links += 1
                    # After all relevant urls found from country websites, find external domain urls
                    else:
                        # Get domain of current url
                        link_domain = urlparse(link_str).netloc

                        # Check if any irrelevant words in url
                        relevance_check = any(site in link_str for site in irrelevant_sites)

                        # If domain is external and url is not irrelevant, add to url_list
                        if url_domain != link_domain and not relevance_check and external_links < external_limit:
                            url_list.append(link_str)
                            external_links += 1
        counter += 1


# Discard the country urls (indexes 0-8) and store all relevant urls in final_urls list
final_urls = url_list[9:]

# Create list for all raw file names
raw_files = []

def scrape_text(relevant_urls):
    """
        Function to loop through all relevant urls, scrape text, and write to file
        Arguments: List of relelvant urls
    """
    # Iterate through all urls
    for index, each_url in enumerate(relevant_urls):
        # HTML parse website and extract content with  <p> tag
        html = request.urlopen(each_url).read().decode('utf8')
        soup = BeautifulSoup(html, 'html.parser')
        paragraph = soup.find_all('p')

        # Get all text
        text = [p.get_text() for p in paragraph]
        text = ' '.join(text)

        # Set filename
        file_name = "scrape_url_" + str(index + 1) + ".txt"

        # Open file and write extracted text to file
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write(text)
            raw_files.append(file_name)

# Call function to scrape text from all relevant urls
scrape_text(final_urls)
print("scrape done")



# Pattern to idenitfy emojis used in text 
emoji = '['\
    u'\U0001F600-\U0001F64F'']+'

# List of cleaned file names
cleaned_files = []

def clean_text(filenames):
    """
        Function to clean scraped text from urls
        Arguments: List of raw text filenames
    """
    # Iterate through all raw text files
    for index, f_name in enumerate(filenames):
        # Create new file name
        new_f_name = "clean_file" + str(index + 1) + ".txt"

        # Open raw text files and read contents
        file = open(f_name, 'r', encoding="utf-8")
        text = file.read()
        file.close()

        # Create string to store cleaned text
        new_text = ''

        # Sentence tokenize text and remove all sentences that contain a date
        sentences = sent_tokenize(text)
        no_dates = [sentence for sentence in sentences if not any(ent.label_ == 'DATE' for ent in nlp(sentence).ents)]
        new_sentences = []

        # Cleaning for all internal domain links
        if index < 15:
            # List of words to discard when cleaning the files
            to_remove = ['consent', 'click', 'cookie', 'privacy', 'plugin', 'data', 'save', 'copyright', 'commission', 'blog', 'kimmie', 'check ', 'chekc', 'pinterest', ' here', 'purchas', 'i love:', 'good dose', 'tips/guides', 'bona-fide', 'sporadic', 'pin ', 'comments closed', 'google', 'link']

            # Remove sentences with meaningless information
            new_sentences = [sentence for sentence in no_dates if not any(word.lower() in sentence.lower() for word in to_remove)]
            
        # Cleaning for all external links
        else:
            # List of words to discard when cleaning the files
            to_remove_external = ['technical issues', 'click', 'cnn', 'warner', 'discovery', 'reserv', 'search', 'booking', 'commission', 'contact', 'terms and conditions', 'privacy', 'data', 'hi', 'top', 'skip', 'page', 'instagram']

            # Remove sentences with meaningless information
            new_sentences = [sentence for sentence in no_dates if not any(word.lower() in sentence.lower() for word in to_remove_external)]
            
            # For last link, removing all text after specific line as following text are comments on a blog and thus are not relevant
            if(index == 19):
                i = new_sentences.index('Let us know in the comments below.')
                new_sentences = new_sentences[:i]
           
        # For all sentences, word tokenize and remove all accents in words
        for i, sent in enumerate(new_sentences):
            temp = []
            for word in word_tokenize(sent):
                if word.isalpha():
                    word = unidecode.unidecode(word)
                temp.append(word)
            new_sentences[i] = ' '.join(temp)

        # Adjust spaces before and after punctuation so cleaned text is more readable using regex
        new_text = ' '.join(new_sentences)
        new_text = re.sub(r'\s([.?!,)”:;%])\s', r'\1 ', new_text)
        new_text = re.sub(r'\s([($“])\s', r' \1', new_text)
        new_text = re.sub(r'[s]\s([’])\s', r'\1 ', new_text)
        new_text = re.sub(r'\s([’])\s', r'\1', new_text)
        new_text = re.sub(r'\s([.])', r'\1', new_text)

        # Remove emojis from text using regex
        new_text = re.sub(emoji, '', new_text).strip()

        # Write cleaned text to a new file for all raw text files
        with open(new_f_name, 'w', encoding="utf-8") as f:
            f.write(''.join(new_text))
            cleaned_files.append(new_f_name)
        
# Call function to clean each raw text file
clean_text(raw_files)
print("cleaning done")


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



def calculate_tf(token_list):
    """
        Calculate tf value
        Arguments: list of tokens
        Output: 
            tf_values - dictionary containing token (key) and tf metric (value) pairs
    """
    tf_values = {}

    # Get count of each token using Counter object
    token_counts = Counter(token_list)

    # Get total token value
    total_tokens = len(token_list)

    # Calculate frequency metric
    for t in token_counts.keys():
        tf_values[t] = token_counts[t] / total_tokens

    return tf_values

def calculate_idf(all_tokens):
    """
        Calculate idf value
        Arguments: corpus of text
        Output: 
            idf_values - dictionary containing token (key) and idf metric (value) pairs
    """
    # Get number of documents
    num_documents = len(all_tokens)

    # Create dictionaries
    idf_values = {}
    idf_count = {}

    # Get count of how many documents token occurs in
    for token_list in all_tokens:
        for token in set(token_list):
            idf_count[token] = idf_count.get(token, 0) + 1
            
    # Calculate idf metric
    for t in idf_count.keys():
        idf_values[t] = math.log((num_documents + 1) / (idf_count[t] + 1))
    
    return idf_values


def calculate_tf_idf(tf_vals, idf_vals):
    """
        Calculate tf-idf value
        Arguments: dictionary of tf values and dictionary of idf values
        Output: 
            tf_idf - dictionary containing token (key) and tf-df metric (value) pairs
    """
    # Create dictionary
    tf_idf = {}

    # Calculate tf-idf metric
    for t in tf_vals.keys():
        tf_idf[t] = tf_vals[t] * idf_vals[t]
    return tf_idf



def extract_terms(clean_files):
    """
        Extract 40 most important terms using tf-idf metric
        Arguments: list cleaned file names
    """
    all_tokens = []
    punctuation = r'[^\w\s]'
    for each_clean_file in clean_files:
        # Open and read each clean file
        with open(each_clean_file, 'r', encoding="utf-8") as f:
            text = f.read().lower()

            # Preprocess text and remove punctuation
            cleaned_tokens = preprocess(text)
            removed_punc = re.sub(punctuation, '', cleaned_tokens).strip()
            all_tokens.append(word_tokenize(removed_punc))
    
    # Calculate idf
    idf = calculate_idf(all_tokens)
    num = 1

    # Calculate tf-idf for each term and print top 40 important terms from sorted dictionary
    for each_list in all_tokens:
        tf_val = calculate_tf(each_list)
        tf_idf = calculate_tf_idf(tf_val, idf)
        sorted_tf_idf = sorted(tf_idf.items(), key=lambda x:x[1], reverse=True)
        print(str(num) + ": " + str(sorted_tf_idf[:2]))
        num += 1

# Call function to get 40 most important terms
extract_terms(cleaned_files)
print("extract done")


# 15 chosen most important words
top_terms = ['bariloche', 'uruguay', 'salta', 'wine', 'iguazu', 'pedro', 'trek', 'natale', 'colombia', 'beagle', 'fitz', 'car', 'cafayate', 'cuidad', 'cordoba']


def get_synonyms(terms):
    """
        Get list of synonyms, hypernyms, and hyponyms of a term and store them in a dictionary
        Arguments: list of terms
        Output: 
            synonyms_dict - dictionary containing term (key) and set of all related words (value) pairs
    """
    synonyms_dict = {}
    for each_term in terms:
        # Create set for each term
        synonyms_dict[each_term] = set()

        # Based on specific term, add related words to synonym set
        if(each_term == 'uruguay'):
            synonyms_dict[each_term].update(['montevideo', 'punta', 'casapueblo', 'piriapolis', 'atlantida', 'isla'])
        if(each_term == 'bariloche'):
            synonyms_dict[each_term].update(['patagonia', 'angostura', 'glacier'])
        if(each_term == 'salta'):
            synonyms_dict[each_term].add('jujuy')
        if(each_term == 'wine'):
            synonyms_dict[each_term].add('food')
        if(each_term == 'iguazu'):
            synonyms_dict[each_term].update(['fall', 'brazil'])
        if(each_term == 'pedro'):
            synonyms_dict[each_term].add('atacama')
        if(each_term == 'trek'):
            synonyms_dict[each_term].update(['hike', 'trail'])
        if(each_term == 'colombia'):
            synonyms_dict[each_term].update(['capurgana', 'sapzurro', 'cielo', 'miel', 'aguacate'])
        if(each_term == 'beagle'):
            synonyms_dict[each_term].update(['ushuaia', 'tierra', 'heli'])
        if(each_term == 'fitz'):
            synonyms_dict[each_term].update(['roy', 'mountain', 'view', 'needle'])
        if(each_term == 'cuidad'):
            synonyms_dict[each_term].add('paraguay')
        if(each_term == 'cordoba'):
            synonyms_dict[each_term].update(['cabido', 'argentina', 'city', 'park'])

        # Iterate through synsets
        for synset in wordnet.synsets(each_term):
            # Clean and add lemma names
            for l in synset.lemmas():
                synonyms_dict[each_term].add(clean_lemma_name(l.name())) 

            # Clean and add hypernym names 
            for hypernym in synset.hypernyms():
                for l in hypernym.lemmas():
                    synonyms_dict[each_term].add(clean_lemma_name(l.name())) 

            # Clean and add hyponym names
            for hyponym in synset.hyponyms():
                for l in hyponym.lemmas():
                    synonyms_dict[each_term].add(clean_lemma_name(l.name())) 
    return synonyms_dict
 
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



# Building knowledge base
def build_kb(terms, clean_files):
    """
        Build knowledge base
        Arguments: lsit of important terms and list of clean file names
        Output: 
            kb - dictionary containing term (key) and all related sentences (value) pairs
    """
    kb = {}
    all_clean = []

    # get synonyms of terms
    all_synonyms = get_synonyms(terms)

    # Iterate through clean files
    for each_file in clean_files:
        with open(each_file, 'r', encoding="utf-8") as f:
            text = f.read()
            all_clean.append(text)
    # Find all related sentences
    for each_term in terms:
        kb[each_term] = find_related_sent(each_term, all_clean, all_synonyms[each_term])
    return kb


def find_related_sent(term, files, synonyms):
    """
        Find sentences that contain the input "term" or synonyms of it
        Arguments: term - word, files - text from file, synonyms - list of words
        Output: 
            related_sentences - list of sentences
    """
    related_sentences = [] 
    for each_file in files:
        for sentence in sent_tokenize(each_file):
            # preprocess sentence
            cleaned_sentence = preprocess(sentence)

            # Check if term or its synonyms is in sentence
            if(term in cleaned_sentence or any(word in word_tokenize(cleaned_sentence) for word in synonyms)):
                related_sentences.append(sentence)
    return related_sentences

# Call function to build knowledge base
knowledge_base = build_kb(top_terms, cleaned_files)
print("kb created")

# Knowledge Base filename
kb_file = "knowledge_base"

# Pickle knowledge base
with open(kb_file, 'wb') as handle:
    pickle.dump(knowledge_base, handle)

print("kb pickled")
