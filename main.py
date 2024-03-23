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
from langchain_groq import ChatGroq

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
groq_llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
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
      goal=f"""Write engaging, insightful articles on {bible_verse} that inform and inspire, blending theology, history, and ethics.""",
      backstory="""As a distinguished journalist celebrated for your profound biblical commentaries, you masterfully blend theological 
                profundity with journalistic lucidity. Your contributions are revered for their unparalleled ability to bridge ancient 
                wisdom with modern dilemmas, rendering the timeless teachings of the Bible profoundly relevant and deeply resonant for 
                today's audience. Your work not only enlightens but also transforms, compelling readers to see the world through a 
                lens that is both ancient and urgently contemporary.""",
      verbose=True,
      llm=groq_llm,
      allow_delegation=True,
      tools=[duckduckgo_search,
             SearchTools.answer_flowise_bible_question,
            ],
  )

  biblical_historian = Agent(
      role="Biblical Historian",
      goal=f"""in short words, illuminate {bible_verse} by exploring its historical, socio-political, and cultural contexts, offering a vivid snapshot of its origins.""",
      backstory="""Your illustrious career as a biblical historian is marked by groundbreaking contributions that have deepened our understanding of the 
                ancient world surrounding the Bible. Through meticulous analysis of archaeological discoveries and historical documents, you masterfully 
                reconstruct the environments and conditions under which the seminal events of the Bible unfolded, breathing life into the ancient narratives
                and connecting the past with the present in a vivid and compelling manner.""",
      verbose=True,
      llm=llm,
      allow_delegation=False,
      tools=[
             SearchTools.search_internet,
            ]
  )

  biblical_linguist = Agent(
      role="Biblical Linguist",
      goal=f"""Unlock the true essence of {bible_verse} in top 5 bullet points by delving into its original language, syntax, and semantics, revealing how words and meanings
                have transformed across time and cultures.""",
      backstory="""As a revered linguist specializing in the ancient languages of the Bible, your journey in deciphering and translating ancient manuscripts
                 has garnered widespread acclaim. With an unparalleled mastery of Hebrew, Aramaic, and Greek, you stand as a luminary in the field, skillfully 
                 bridging millennia to connect the ancient world with contemporary insight. Your profound understanding not only brings ancient texts to life
                 but also unlocks their deepest meanings, offering a key to the timeless wisdom they contain for a modern audience.""",
      verbose=True,
      llm=llm,
      allow_delegation=False,
      tools=[
            SearchTools.search_internet,
            ]
  )

  task1=Task(
      description=f"""Start by quoting {bible_verse}. Then, undertake a deep exploration to write an enlightening article on {bible_verse}, delving into its theological
                     depth, historical context, and ethical dimensions. Collaborate with a Biblical Historian to gain profound historical perspectives and consult a 
                     Biblical Linguist to unravel linguistic nuances. Aim to create a narrative that not only informs but also spiritually uplifts the reader, merging 
                     meticulous research, engaging storytelling, and thoughtful reflection for a transformative experience.""",
      agent=biblical_journalist,
  )

  task2=Task(
      description=f"""Explore the history behind {bible_verse}, delving into archaeological discoveries, societal norms, and the period's politics to illuminate its context.
                     Work alongside the Biblical Journalist to enrich their article with historical depth, and assist the Biblical Linguist with insights for their analysis. 
                     Your expertise will vividly spotlight the verse's authentic historical setting.""",
      agent=biblical_historian,
  )

  task3=Task(
      description=f"""Dive deep into the linguistic essence of {bible_verse}, dissecting the original languages to uncover syntactical, semantic, and historical linguistic
                     layers that shape its interpretation. Collaborate with the Biblical Historian to ground your linguistic insights in historical truth, and guide the 
                     Biblical Journalist to weave an article that stands out for its precision and richness. Your linguistic acumen is crucial in revealing the
                     multi-dimensional meanings and making them accessible to today's readers.""",
      agent=biblical_linguist,
  )

  product_crew = Crew(
      agents=[biblical_journalist,biblical_historian,biblical_linguist],
      tasks=[task3,task2,task1],
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
