import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util










science_questions = [
    
    {"question": "What is the speed of light?", "category": "Science"},
    {"question": "Define the term biodiversity.", "category": "Science"},
    {"question": "What is the process of osmosis?", "category": "Science"},
    {"question": "Name the parts of the water cycle.", "category": "Science"},
    {"question": "What is the primary function of red blood cells?", "category": "Science"},
    {"question": "What is the formula for photosynthesis?", "category": "Science"},
    {"question": "What is an exoskeleton?", "category": "Science"},
    {"question": "What are the different types of clouds?", "category": "Science"},
    {"question": "Define the greenhouse effect.", "category": "Science"},
    {"question": "What are Newton’s three laws of motion?", "category": "Science"},
    {"question": "What is the function of mitochondria in a cell?", "category": "Science"},
    {"question": "What is the difference between a comet and an asteroid?", "category": "Science"},
    {"question": "What is the unit of measurement for energy?", "category": "Science"},
    {"question": "What is the role of decomposers in an ecosystem?", "category": "Science"},
    {"question": "What causes the aurora borealis?", "category": "Science"},
    {"question": "What is the law of conservation of mass?", "category": "Science"},
    {"question": "Define the term “black hole.”", "category": "Science"},
    {"question": "What is genetic engineering?", "category": "Science"},
    {"question": "Name the main types of renewable energy.", "category": "Science"},
    {"question": "What is the process of pollination?", "category": "Science"},
    {"question": "What are the layers of the atmosphere?", "category": "Science"},
    {"question": "Define a chemical equation.", "category": "Science"},
    {"question": "What is the function of the ozone layer?", "category": "Science"},
    {"question": "What is a symbiotic relationship?", "category": "Science"},
    {"question": "What is the pH scale used to measure?", "category": "Science"},
    {"question": "What are the characteristics of mammals?", "category": "Science"},
    {"question": "What is an element in the periodic table?", "category": "Science"},
    {"question": "Define absolute zero.", "category": "Science"},
    {"question": "What is the difference between weather and climate?", "category": "Science"},
    {"question": "What is the function of the nervous system?", "category": "Science"},
    {"question": "What is a nebula?", "category": "Science"},
    {"question": "Name three types of fossil fuels.", "category": "Science"},
    {"question": "What is plate tectonics?", "category": "Science"},
    {"question": "What is the difference between a virus and bacteria?", "category": "Science"},
    {"question": "What is the process of sedimentation?", "category": "Science"},
    {"question": "What is the function of roots in plants?", "category": "Science"},
    {"question": "Define the term “ecosystem services.”", "category": "Science"},
    {"question": "What is a gene?", "category": "Science"},
    {"question": "What are the properties of metals?", "category": "Science"},
    {"question": "What is the periodic table?", "category": "Science"},
    {"question": "What is the difference between mass and weight?", "category": "Science"},
    {"question": "Define the Big Bang Theory.", "category": "Science"},
    {"question": "What is sound energy?", "category": "Science"},
    {"question": "What are the properties of acids?", "category": "Science"},
    {"question": "What is a heterogeneous mixture?", "category": "Science"},
    {"question": "What is the role of nitrogen in the environment?", "category": "Science"},
    {"question": "What is kinetic energy?", "category": "Science"},
    {"question": "What is the process of fermentation?", "category": "Science"},
    {"question": "What are the different phases of the Moon?", "category": "Science"},
    {"question": "What is the difference between a physical change and a chemical change?", "category": "Science"},
    {"question": "What is photosynthesis?", "category": "Science"},
    {"question": "What is gravity?", "category": "Science"},
    {"question": "What is the boiling point of water?", "category": "Science"},
    {"question": "What are the three states of matter?", "category": "Science"},
    {"question": "Define atom.", "category": "Science"},
    {"question": "What is the function of the heart?", "category": "Science"},
    {"question": "What is a cell membrane?", "category": "Science"},
    {"question": "What is the role of chlorophyll?", "category": "Science"},
    {"question": "What is an ecosystem?", "category": "Science"},
    {"question": "What are inherited traits?", "category": "Science"},
    {"question": "What is a chemical reaction?", "category": "Science"},
    {"question": "Name three organs in the human body.", "category": "Science"},
    {"question": "What is friction?", "category": "Science"},
    {"question": "What causes day and night?", "category": "Science"},
    {"question": "What is the solar system?", "category": "Science"},
    {"question": "What is an element?", "category": "Science"},
    {"question": "What is energy?", "category": "Science"},
    {"question": "What is the water cycle?", "category": "Science"},
    {"question": "Name three types of rocks.", "category": "Science"},
    {"question": "What is a habitat?", "category": "Science"},
    {"question": "What are renewable resources?", "category": "Science"},
    {"question": "What is the greenhouse effect?", "category": "Science"},
    {"question": "What are the layers of the Earth?", "category": "Science"},
    {"question": "What is DNA?", "category": "Science"},
    {"question": "Define a molecule.", "category": "Science"},
    {"question": "What is magnetism?", "category": "Science"},
    {"question": "What is evaporation?", "category": "Science"},
    {"question": "What are chromosomes?", "category": "Science"},
    {"question": "What is a virus?", "category": "Science"},
    {"question": "What is a chemical bond?", "category": "Science"},
    {"question": "What is a volcano?", "category": "Science"},
    {"question": "What is a tsunami?", "category": "Science"},
    {"question": "Define biodiversity.", "category": "Science"},
    {"question": "What is a comet?", "category": "Science"},
    {"question": "What are the planets in the solar system?", "category": "Science"},
    {"question": "What are natural disasters?", "category": "Science"},
    {"question": "What is genetic mutation?", "category": "Science"},
    {"question": "What is fossilization?", "category": "Science"},
    {"question": "What causes earthquakes?", "category": "Science"},
    {"question": "What is the water cycle?", "category": "Science"},
    {"question": "What is photosynthesis?", "category": "Science"},
    {"question": "What is gravity?", "category": "Science"},
    {"question": "What is the boiling point of water?", "category": "Science"},
    {"question": "What are the three states of matter?", "category": "Science"},
    {"question": "Define atom.", "category": "Science"},
    {"question": "What is the function of the heart?", "category": "Science"},
    {"question": "What is a cell membrane?", "category": "Science"},
    {"question": "What is the role of chlorophyll?", "category": "Science"},
    {"question": "What is an ecosystem?", "category": "Science"},
    {"question": "What are inherited traits?", "category": "Science"},
    {"question": "What is a chemical reaction?", "category": "Science"},
    {"question": "Name three organs in the human body.", "category": "Science"},
    {"question": "What is friction?", "category": "Science"},
    {"question": "What causes day and night?", "category": "Science"},
    {"question": "What is the solar system?", "category": "Science"},
    {"question": "What is an element?", "category": "Science"},
    {"question": "What is energy?", "category": "Science"},
]



math_questions = [
    
    {"question": "Solve 2x + 3 = 7", "category": "Math"},
    {"question": "What is the square root of 25?", "category": "Math"},
    {"question": "What is 12 multiplied by 8?", "category": "Math"},
    {"question": "What is the area of a circle?", "category": "Math"},
    {"question": "Solve: 3x - 2 = x + 4", "category": "Math"},
    {"question": "What is the formula for perimeter of a rectangle?", "category": "Math"},
    {"question": "What is a prime number?", "category": "Math"},
    {"question": "What is the value of pi?", "category": "Math"},
    {"question": "Find x if x^2 = 64", "category": "Math"},
    {"question": "What is the mean of 5, 10, 15?", "category": "Math"},
    {"question": "Define integer.", "category": "Math"},
    {"question": "What is a fraction?", "category": "Math"},
    {"question": "What is a factor?", "category": "Math"},
    {"question": "What is a multiple?", "category": "Math"},
    {"question": "Convert 0.75 to a fraction.", "category": "Math"},
    {"question": "Add: 3/4 + 2/4", "category": "Math"},
    {"question": "Subtract: 9 - 4", "category": "Math"},
    {"question": "What is the slope of a line?", "category": "Math"},
    {"question": "What is the Pythagorean theorem?", "category": "Math"},
    {"question": "What is a composite number?", "category": "Math"},
    {"question": "Define a polygon.", "category": "Math"},
    {"question": "What is a histogram?", "category": "Math"},
    {"question": "Solve: 5x + 3 = 18", "category": "Math"},
    {"question": "What is the formula for the area of a triangle?", "category": "Math"},
    {"question": "Find the percentage of 50 out of 200.", "category": "Math"},
    {"question": "What are parallel lines?", "category": "Math"},
    {"question": "What is the median of 3, 8, 6, 12, 10?", "category": "Math"},
    {"question": "What is a quadratic equation?", "category": "Math"},
    {"question": "Solve: x^2 - 4x + 4 = 0", "category": "Math"},
    {"question": "Solve 2x + 3 = 7", "category": "Math"},
    {"question": "What is the square root of 25?", "category": "Math"},
    {"question": "What is 12 multiplied by 8?", "category": "Math"},
    {"question": "What is the area of a circle?", "category": "Math"},
    {"question": "Solve: 3x - 2 = x + 4", "category": "Math"},
    {"question": "What is the formula for perimeter of a rectangle?", "category": "Math"},
    {"question": "What is a prime number?", "category": "Math"},
    {"question": "What is the value of pi?", "category": "Math"},
    {"question": "Find x if x^2 = 64", "category": "Math"},
    {"question": "What is the mean of 5, 10, 15?", "category": "Math"},
    {"question": "Define integer.", "category": "Math"},
    {"question": "What is a fraction?", "category": "Math"},
    {"question": "What is a factor?", "category": "Math"},
    {"question": "What is a multiple?", "category": "Math"},
    {"question": "Convert 0.75 to a fraction.", "category": "Math"},
    {"question": "Add: 3/4 + 2/4", "category": "Math"},
    {"question": "Subtract: 9 - 4", "category": "Math"},
    {"question": "What is the sum of 34 and 56?", "category": "Math"},
    {"question": "What is 144 divided by 12?", "category": "Math"},
    {"question": "Simplify: 6x + 3x.", "category": "Math"},
    {"question": "What is the square root of 81?", "category": "Math"},
    {"question": "What is the value of 7 factorial?", "category": "Math"},
    {"question": "What is the perimeter of a square with side length 8?", "category": "Math"},
    {"question": "Solve for x: 5x = 25.", "category": "Math"},
    {"question": "What is a reciprocal?", "category": "Math"},
    {"question": "Define an irrational number.", "category": "Math"},
    {"question": "What is the area of a rectangle with dimensions 9 by 4?", "category": "Math"},
    {"question": "What is 0.5 as a percentage?", "category": "Math"},
    {"question": "What are consecutive numbers?", "category": "Math"},
    {"question": "What is the LCM of 12 and 15?", "category": "Math"},
    {"question": "Find the GCD of 18 and 24.", "category": "Math"},
    {"question": "What is a linear equation?", "category": "Math"},
    {"question": "What is an arithmetic sequence?", "category": "Math"},
    {"question": "Solve for y: y/4 = 6.", "category": "Math"},
    {"question": "What are perpendicular lines?", "category": "Math"},
    {"question": "What is the difference between radius and diameter?", "category": "Math"},
    {"question": "What are similar triangles?", "category": "Math"},
    {"question": "What is the value of x in x^2 = 121?", "category": "Math"},
    {"question": "What is the slope-intercept form of a line?", "category": "Math"},
    {"question": "What is the average of 10, 15, and 20?", "category": "Math"},
    {"question": "Convert 1/4 to a decimal.", "category": "Math"},
    {"question": "What is the cube of 3?", "category": "Math"},
    {"question": "What is the formula for the circumference of a circle?", "category": "Math"},
    {"question": "What is a polygon with five sides called?", "category": "Math"},
    {"question": "What is the probability of rolling a six on a dice?", "category": "Math"},
    {"question": "What is a scatter plot?", "category": "Math"},
    {"question": "Simplify: 15x - 7x.", "category": "Math"},
    {"question": "Solve for z: 3z + 7 = 16.", "category": "Math"},
    {"question": "What is an exponential function?", "category": "Math"},
    {"question": "What is a histogram used for?", "category": "Math"},
    {"question": "What are quadrilaterals?", "category": "Math"},
    {"question": "What is the formula for finding simple interest?", "category": "Math"},
    {"question": "Solve: 2x - 3 = 7.", "category": "Math"},
    {"question": "What is a coefficient?", "category": "Math"},
    {"question": "What is the distributive property?", "category": "Math"},
    {"question": "What is a variable in algebra?", "category": "Math"},
    {"question": "Define the term ‘prime factorization.’", "category": "Math"},
    {"question": "What are vertical angles?", "category": "Math"},
    {"question": "What is the difference between direct and inverse variation?", "category": "Math"},
    {"question": "What is the formula for the area of a trapezoid?", "category": "Math"},
    {"question": "What is the decimal representation of 3/5?", "category": "Math"},
    {"question": "What is the square of 12?", "category": "Math"},
    {"question": "What is the total of the interior angles of a triangle?", "category": "Math"},
    {"question": "What is the standard form of a quadratic equation?", "category": "Math"},
    {"question": "What is the length of the hypotenuse in a right triangle?", "category": "Math"},
    {"question": "What is the formula for compound interest?", "category": "Math"},
    {"question": "What is the difference between mean and median?", "category": "Math"},
    
    
]

english_questions = [
    {"question": "Define a noun and give examples.", "category": "English"},
    {"question": "What is the difference between a verb and an adjective?", "category": "English"},
    {"question": "What are proper nouns?", "category": "English"},
    {"question": "What is the past tense of 'write'?", "category": "English"},
    {"question": "What is the plural form of 'mouse'?", "category": "English"},
    {"question": "What are synonyms and antonyms?", "category": "English"},
    {"question": "What is the function of a preposition?", "category": "English"},
    {"question": "What is the structure of a compound sentence?", "category": "English"},
    {"question": "What is a possessive pronoun?", "category": "English"},
    {"question": "What is the meaning of an idiom?", "category": "English"},
    {"question": "What are homophones? Provide examples.", "category": "English"},
    {"question": "What is an interrogative sentence?", "category": "English"},
    {"question": "What is the difference between active and passive voice?", "category": "English"},
    {"question": "What is the role of conjunctions in a sentence?", "category": "English"},
    {"question": "What is a narrative essay?", "category": "English"},
    {"question": "Define a metaphor and give examples.", "category": "English"},
    {"question": "What is an exclamatory sentence?", "category": "English"},
    {"question": "What are prefixes and suffixes?", "category": "English"},
    {"question": "What is direct and indirect speech?", "category": "English"},
    {"question": "What are transitive and intransitive verbs?", "category": "English"},
    {"question": "What is the main idea of a story?", "category": "English"},
    {"question": "What is the difference between a phrase and a clause?", "category": "English"},
    {"question": "What is subject-verb agreement?", "category": "English"},
    {"question": "What is the plural of 'child'?", "category": "English"},
    {"question": "What are descriptive adjectives?", "category": "English"},
    {"question": "What is figurative language?", "category": "English"},
    {"question": "Define personification and give examples.", "category": "English"},
    {"question": "What is the meaning of a simile?", "category": "English"},
    {"question": "What are rhyming words?", "category": "English"},
    {"question": "What is the purpose of punctuation marks?", "category": "English"},
    {"question": "What is a declarative sentence?", "category": "English"},
    {"question": "What are determiners in grammar?", "category": "English"},
    {"question": "What are coordinating conjunctions?", "category": "English"},
    {"question": "What is a question tag?", "category": "English"},
    {"question": "What is an oxymoron?", "category": "English"},
    {"question": "What is the function of an adverb?", "category": "English"},
    {"question": "What is a compound word? Provide examples.", "category": "English"},
    {"question": "What is an abstract noun?", "category": "English"},
    {"question": "What is an imperative sentence?", "category": "English"},
    {"question": "Define a hyperbole and give examples.", "category": "English"},
    {"question": "What is the difference between literal and figurative meaning?", "category": "English"},
    {"question": "What is the function of a predicate in a sentence?", "category": "English"},
    {"question": "What is an article in English grammar?", "category": "English"},
    {"question": "What is the difference between countable and uncountable nouns?", "category": "English"},
    {"question": "What is an interjection?", "category": "English"},
    {"question": "What is the difference between synonyms and homonyms?", "category": "English"},
    {"question": "What is parallelism in writing?", "category": "English"},
    {"question": "What is an appositive?", "category": "English"},
    {"question": "What is the use of a semicolon in a sentence?", "category": "English"},
        {"question": "What is a noun?", "category": "English"},
    {"question": "What is a verb?", "category": "English"},
    {"question": "What is an adjective?", "category": "English"},
    {"question": "What is the past tense of 'go'?", "category": "English"},
    {"question": "What is a pronoun?", "category": "English"},
    {"question": "What is an adverb?", "category": "English"},
    {"question": "What is a sentence?", "category": "English"},
    {"question": "Give an example of a preposition.", "category": "English"},
    {"question": "What is a synonym for 'happy'?", "category": "English"},
    {"question": "What is an antonym for 'tall'?", "category": "English"},
    {"question": "Use 'because' in a sentence.", "category": "English"},
    {"question": "What is a conjunction?", "category": "English"},
    {"question": "Identify the subject in this sentence.", "category": "English"},
    {"question": "What is the plural of 'child'?", "category": "English"},
    {"question": "Define a compound sentence.", "category": "English"},
    {"question": "What is punctuation?", "category": "English"},
    {"question": "What is a question tag?", "category": "English"},
    {"question": "What is a narrative?", "category": "English"},
    {"question": "What is figurative language?", "category": "English"},
    {"question": "Define a metaphor.", "category": "English"},
    {"question": "What is a noun?", "category": "English"},
    {"question": "What is a verb?", "category": "English"},
    {"question": "What is an adjective?", "category": "English"},
    {"question": "What is the past tense of 'go'?", "category": "English"},
    {"question": "What is a pronoun?", "category": "English"},
    {"question": "What is an adverb?", "category": "English"},
    {"question": "What is a sentence?", "category": "English"},
    {"question": "Give an example of a preposition.", "category": "English"},
    {"question": "What is a synonym for 'happy'?", "category": "English"},
    {"question": "What is an antonym for 'tall'?", "category": "English"},
    {"question": "Use 'because' in a sentence.", "category": "English"},
    {"question": "What is a conjunction?", "category": "English"},
    {"question": "Identify the subject in this sentence.", "category": "English"},
    {"question": "What is the plural of 'child'?", "category": "English"},
    {"question": "Define a compound sentence.", "category": "English"},
    {"question": "What is punctuation?", "category": "English"},
    {"question": "What is a question tag?", "category": "English"},
]


data=science_questions + math_questions + english_questions

df=pd.DataFrame(data)

x =df['question']
y =df['category']


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, stratify=y,random_state=42)


model=make_pipeline(TfidfVectorizer(),LogisticRegression(class_weight="balanced"))

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))






def handle_user_input(user_input):
    predicted_category = model.predict([user_input])[0]
    if predicted_category == "Science":
        return handle_science_question(user_input)
    elif predicted_category == "Math":
        return handle_math_question(user_input)
    elif predicted_category == "English":
        return handle_english_question(user_input)
    else:
        return" sorry  i don't know the answer to this question"
    
    

def handle_science_question(question):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, science_embeddings)[0]
    best_match_idx = scores.argmax().item()
    return science_kb[best_match_idx]["a"]
    
      
     
def handle_math_question(question):
   try:
        if "solve" in question.lower():
            # Example: "Solve 2x + 3 = 7"
            return "This looks like an equation. I'm working on solving it..."
        elif any(op in question for op in ["+", "-", "*", "/"]):
            return str(eval(question.split(":")[-1]))  # crude eval; not recommended for real use
        else:
            return "Can you rephrase your math question?"
   except:
        return "Sorry, I couldn't solve that equation yet."
    
    
    
def handle_english_question(question):
    if "noun" in question.lower():
        return "A noun is a person, place, thing, or idea."
    elif "verb" in question:
        return "A verb is an action word."
    else:
        return "That's an English question! I'm still learning to answer more of them."        
    
    

    
embedder = SentenceTransformer('all-MiniLM-L6-v2')    


science_kb = [
    {"q": "What is photosynthesis?", "a": "Photosynthesis is how plants make food using sunlight."},
    {"q": "What is gravity?", "a": "Gravity is the force that attracts objects towards the earth."},
    {"q": "What is an atom?", "a": "An atom is the smallest unit of matter."}
]

# Encode all questions
science_embeddings = embedder.encode([item["q"] for item in science_kb], convert_to_tensor=True)



user_input = input("Ask me a question: ")
response = handle_user_input(user_input)
print("AI Tutor:", response)
    