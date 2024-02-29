from flask import Flask, request, jsonify, render_template
from flask_mail import Mail, Message
import os
import requests
import json
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.tools import Tool
from langchain.tools import tool
from groq import Groq

# Initialize Flask app
app = Flask(__name__)

# Fetch mail configuration from environment variables
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))  # Defaulting to 587 if not set
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'  # Converting string to boolean
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False') == 'True'  # Converting string to boolean

mail = Mail(app)

# Fetch API keys from environment variables
serper_api_key = os.getenv("SERPER_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY", "default-key")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API keys are available
if not serper_api_key:
    raise ValueError("SERPER_API_KEY is not set in the environment variables.")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initializing dependencies
groq_llm = Groq()
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.5, google_api_key=google_api_key)
duckduckgo_search = DuckDuckGoSearchRun()

# SearchTools class as defined in your notebook
class SearchTools:
    @tool("Search the internet")
    def search_internet(bible_verse):
        """Search the internet about a given topic and return relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": bible_verse})
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            results = response.json().get('organic', [])
            return results
        else:
            return "Failed to search the internet."

    @tool("Search for Places on the internet")
    def search_places(bible_verse):
        """Search for Places on a geographical context of a bible verse and return relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/places"
        payload = json.dumps({"q": bible_verse})
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            results = response.json().get('places', [])
            return SearchTools.format_results(results, top_result_to_return)
        else:
            return "Failed to search places."

    @tool("Search for a topic related to the Bible & Christianity")
    def answer_flowise_bible_question(bible_verse):
        """Ask any Bible related question and get relevant results."""
        top_result_to_return = 1
        url = "https://flow.koltelecom.com/api/v1/prediction/56c19c1f-1e29-436b-b322-df00eb66a998"
        payload = json.dumps({"question": bible_verse})
        response = requests.post(url, json={"question": bible_verse })
        if response.status_code == 200:
            results = response.json().get('answer', [])
            return SearchTools.format_results(results, top_result_to_return)
        else:
            return "Failed to get result from Flowise."
        
    @staticmethod
    def format_results(results, limit):
        formatted = []
        for result in results[:limit]:
            formatted.append('\n'.join([
                f"Title: {result.get('title', 'N/A')}",
                f"Link: {result.get('link', 'N/A')}",
                f"Snippet: {result.get('snippet', 'N/A')}",
                "\n-----------------"
            ]))
        return '\n'.join(formatted)
    

def create_crewai_setup(bible_verse):
  biblical_journalist = Agent(
      role="Biblical Journalist",
      goal=f"""To craft deeply engaging and spiritually insightful articles on {bible_verse}, delving into its theological implications, historical relevance, and ethical teachings.
                Your task is to weave a narrative that not only informs but also inspires readers, drawing on diverse theological insights and scholarly perspectives.""",
      backstory="""As an acclaimed journalist renowned for thought-provoking biblical commentaries, you merge theological depth with journalistic clarity. Your work is known for its 
                    ability to connect ancient texts with contemporary issues, making the Bible relevant and accessible to a modern audience.""",
      verbose=True,
      llm=groq_llm,
      allow_delegation=True,
      tools=[duckduckgo_search,
             SearchTools.answer_flowise_bible_question,
            ],
  )

  biblical_historian = Agent(
      role="Biblical Historian",
      goal=f"""To meticulously analyze {bible_verse} within its historical setting, shedding light on its socio-political and cultural contexts. Your expertise in ancient
                Near Eastern history and culture is crucial in uncovering the subtleties and nuances hidden in the text, offering a vivid picture of the era it was written in.""",
      backstory="""Your career as a historian is distinguished by your contributions to understanding the world of the Bible. With a focus on archeological findings and historical texts,
                    you bring to life the settings and circumstances in which biblical events occurred.""",
      verbose=True,
      llm=groq_llm,
      allow_delegation=True,
      tools=[SearchTools.search_places,
             SearchTools.search_internet,
            ]
  )

  biblical_linguist = Agent(
      role="Biblical Linguist",
      goal=f"""To dissect and interpret the linguistic intricacies of {bible_verse}, examining its original language, syntax, and semantics.
                Your task is to illuminate the verse's authentic meanings, exploring how linguistic evolution and cultural context have shaped its interpretation over time.""",
      backstory="""An esteemed linguist, you specialize in ancient biblical languages. Your work in deciphering ancient manuscripts and translating them
                    accurately has been widely acknowledged. Your deep understanding of Hebrew, Aramaic, and Greek languages enables you to bridge
                    the gap between ancient texts and modern understanding.""",
      verbose=True,
      llm=llm,
      allow_delegation=True,
      tools=[
            SearchTools.search_internet,
            ]
  )

  task1=Task(
      description=f"""Firstly recite: {bible_verse}. Then embark on an investigative journey to craft an illuminating article on {bible_verse}. Dive deep into its theological significance, historical relevance, and ethical implications.
                  Consult with the Biblical Historian for in-depth historical insights, and engage with the Biblical Linguist to decode linguistic subtleties.
                  Your goal is to produce a piece that not only educates but also spiritually enriches the reader, by blending rigorous research, creative storytelling, and reflective analysis.""",
      agent=biblical_journalist,
  )

  task2=Task(
      description=f"""Your task is to unravel the historical tapestry surrounding {bible_verse}. Examine the archaeological findings, societal norms, and political climate of the period to shed
                  light on the context of the verse. Collaborate with the Biblical Journalist to provide a rich, historical backdrop for their article, and offer insights to the Biblical Linguist
                  to aid in their linguistic analysis. Your comprehensive understanding of the era’s history will illuminate the verse's true place in time.""",
      agent=biblical_historian,
  )

  task3=Task(
      description=f"""Undertake a thorough linguistic analysis of {bible_verse}, exploring the nuances of its original languages. Assess how its syntax, semantics, and historical linguistic variations
                  influence its interpretation. Work closely with the Biblical Historian to align your linguistic findings with historical contexts and assist the Biblical Journalist in crafting an
                  article that resonates with accuracy and depth. Your expertise is key to unlocking the layered meanings and translating them for a contemporary audience.""",
      agent=biblical_linguist,
  )

  product_crew = Crew(
      agents=[biblical_journalist,biblical_historian,biblical_linguist],
      tasks=[task1],
      verbose=2,
      process=Process.sequential,
  )

  crew_result = product_crew.kickoff()
  return crew_result

# Function to run Crew AI
def run_crewai(bible_verse):
    crew_result = create_crewai_setup(bible_verse)
    return crew_result

# Flask route for processing a Bible verse using Crew AI

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        bible_verse = request.form.get('verse')
        if bible_verse:
            result = run_crewai(bible_verse)
    return render_template('index.html', result=result)

    
@app.route('/process_verse', methods=['POST'])
def process_verse():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    book = data.get('book')
    chapter = data.get('chapter')
    verse = data.get('verse')
    email = data.get('email')

    if not all([book, chapter, verse, email]):
        return jsonify({"error": "Book, chapter, verse or email not provided"}), 400

    try:
        # Combine book, chapter, and verse for Crew AI processing
        bible_verse = f"{book} {chapter}:{verse}"
        result = run_crewai(bible_verse)

        # Prepare and send the email
        msg = Message("Your Bible Verse Result",
                      sender='verse.insights@gmail.com',
                      recipients=[email])
        msg.body = 'Here is the result of your request: \n\n' + str(result)
        mail.send(msg)

        return jsonify({"message": "Email sent successfully with the results."})
    except Exception as e:
        # Log the exception and inform the user
        app.logger.error("Error processing verse or sending email", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request."}), 500


if __name__ == '__main__':
    app.run(debug=True)
