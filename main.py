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

# Check if the API keys are available
if not serper_api_key:
    raise ValueError("SERPER_API_KEY is not set in the environment variables.")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initializing dependencies
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

    @tool("Search for anything on the Bible & Christianity")
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
      goal=f"""Write very high quality insightful articles and research papers which are worthy to publish on Biblical subjects, characters and theology.
               Should collect the necessary data and prepare it for verse: {bible_verse}""",
      backstory="""Experienced in journalism with a specialization in biblical topics, combining the expertise in theology, history, and journalism to
                    interpret and communicate the Bible's historical, cultural, and religious significance. Work involves
                    in-depth research, ethical reporting, and engagement with a network of scholars and experts in these domains.""",
      verbose=True,
      llm=llm,
      allow_delegation=True,
      tools=[duckduckgo_search,
             SearchTools.answer_flowise_bible_question,
            ],
  )

  biblical_historian = Agent(
      role="Biblical Historian",
      goal="""to investigate the historical context of the Bible, unravel the cultural, social, and political landscapes of the era,
                and provide an objective analysis of biblical events and figures, enhancing the understanding of the scripture's place in history.""",
      backstory="""pursues in-depth studies in history and theology, focusing on ancient cultures, languages, and archaeological evidence to objectively
                    analyze and contextualize the events, figures, and narratives within the Bible in their historical setting.""",
      verbose=True,
      llm=llm,
      allow_delegation=False,
      tools=[SearchTools.search_places,
             SearchTools.search_internet,
            ]
  )

  biblical_linguist = Agent(
      role="Biblical Linguist",
      goal=f""" to analyze and interpret the original languages of the Bible, such as Hebrew, Aramaic, and Greek,
                to understand scriptural texts in their authentic context and to elucidate their meanings, nuances,
                and linguistic evolution over time.""",
      backstory="""Highly skilled & extensive studies in ancient languages and linguistics, followed by specialization
                   in the languages of the Bible. Focus on text analysis, cultural context, and historical language development, contributing to scholarly research
                   and more accurate translations of biblical texts.""",
      verbose=True,
      llm=llm,
      allow_delegation=False,
      tools=[
            SearchTools.search_internet,
            ]
  )

  task1=Task(
      description=f"""As a biblical journalist preparing an article/blog on {bible_verse}, engage in a multi-disciplinary approach: consult Biblical Historian for insights into
                   the verse's historical and cultural background, collaborate with Biblical Linguist to understand linguistic nuances and implications, and identify the author
                    and audience of the verse. Gather extensive information, incorporating varied perspectives, especially from expert peers, to ensure a
                    deeply researched, well-analyzed, and comprehensive exploration of {bible_verse} in your content.""",
      agent=biblical_journalist,
  )

  task2=Task(
      description=f"""analyze the historical context, examining archaeological, cultural, and socio-political aspects of the period,
                      and cross-reference contemporary historical sources to provide a comprehensive understanding of the verse: {bible_verse}, its setting and significance.""",
      agent=biblical_historian,
  )

  task3=Task(
      description=f"""analyze the original language, syntax, and semantics of {bible_verse}, considering linguistic variations and historical usage, to interpret its meaning, nuances,
                      and potential translation intricacies within its cultural and historical context.""",
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
    verse = request.form['verse']
    email = request.form['email']
    result = run_crewai(verse)  # This is your function to process the verse

    # Preparing the email message
    
    return jsonify({"message": "Email processing started"})
    
    msg = Message("Your Bible Verse Result", 
                  sender='sdcharly@gmail.com', 
                  recipients=[email])
    msg.body = 'Here is the result of your request: \n\n' + str(result)

    # Sending the email
    mail.send(msg)

    return "The result has been sent to your email."

if __name__ == '__main__':
    app.run(debug=True)
