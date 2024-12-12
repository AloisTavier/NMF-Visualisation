#NMF implementation
import json
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
import os

size_directory = 0
for file in os.listdir("Texts"):
    size_directory += 1
print(f"size of directory = {size_directory}")
# Document collection with texts on Python, piano music, and food
document_vector2 = [
    # 1 Python text
    "Python is one of the most versatile programming languages available today. It is widely used in fields like web development, data science, and artificial intelligence. The language's simplicity and readability make it an excellent choice for beginners and professionals alike. With a rich ecosystem of libraries such as NumPy, Pandas, and TensorFlow, Python has become indispensable for tasks involving data manipulation, statistical analysis, and machine learning. Additionally, its cross-platform compatibility allows developers to run Python programs on multiple operating systems seamlessly.",

    # 2 Piano music text
    "The piano is a beloved instrument that has captivated audiences for centuries. Its versatility allows it to be used in genres ranging from classical and jazz to pop and rock. Playing the piano requires coordination between both hands, as they often perform independent melodies or harmonies. Learning the piano can enhance cognitive abilities and improve hand-eye coordination. The works of legendary composers such as Beethoven, Chopin, and Debussy showcase the instrument’s incredible range and expressive potential.",

    # 3 Sushi text
    "Sushi is a traditional Japanese dish that has gained immense popularity worldwide. It typically consists of vinegared rice combined with a variety of ingredients, including raw fish, vegetables, and seaweed. Sushi comes in many forms, such as nigiri, maki, and sashimi, each offering a unique taste and texture. Crafting sushi is considered an art form, requiring precise techniques and fresh ingredients. Sushi chefs, or itamae, undergo years of rigorous training to master their craft.",

    # 4 Python text
    "Python’s use in web development is facilitated by frameworks such as Django and Flask. These tools simplify the process of building robust web applications. With Django, developers can create scalable and secure websites quickly, while Flask provides flexibility for lightweight and customizable projects. The availability of third-party libraries further enhances Python's capability in web development. For instance, developers can easily integrate authentication systems, databases, and APIs into their applications.",

    # 5 Piano music text
    "One of the most enchanting aspects of piano music is its ability to convey emotions. A slow and gentle melody can evoke a sense of calm, while fast-paced and intense compositions can energize the listener. The dynamic range of the piano allows it to be both soft and thunderous, making it ideal for storytelling through music. Pieces like Chopin's Nocturnes or Liszt’s Hungarian Rhapsodies demonstrate the instrument's emotional depth and complexity.",

    # 6 Sushi text
    "The cultural significance of sushi in Japan is profound. It is more than just food; it represents a connection to Japanese traditions and values. Sushi is often served during celebrations and ceremonies, emphasizing its role in bringing people together. The meticulous presentation of sushi reflects the Japanese aesthetic principle of ‘wabi-sabi,’ which values simplicity and harmony. In modern times, sushi has evolved to incorporate international flavors, creating fusion dishes that appeal to diverse tastes.",

    # 7 Python text
    "Python excels in data analysis, thanks to libraries like Pandas, Matplotlib, and Seaborn. These tools enable users to manipulate datasets, visualize trends, and conduct statistical analyses with ease. For example, Pandas simplifies working with data frames, while Matplotlib creates stunning visualizations. Python's integration with Jupyter Notebooks makes it a preferred choice for data scientists, as they can write code, run analyses, and share findings in an interactive and collaborative environment.",

    # 8 Sushi text
    "Sushi is a healthy and nutrient-rich meal when prepared with fresh ingredients. The combination of rice, fish, and vegetables provides a balanced intake of carbohydrates, protein, and essential vitamins. For instance, raw fish like salmon and tuna are excellent sources of omega-3 fatty acids, which support heart and brain health. Vegetarian options such as cucumber and avocado rolls offer a delicious alternative for those who prefer plant-based diets. With low calorie counts and high nutritional value, sushi is both a tasty and wholesome choice.",

    # 9 Piano music text
    "The piano plays a central role in music education. Many students start their musical journey with the piano because of its clear visual layout and versatility. Learning to play the piano provides a strong foundation in music theory, as it teaches concepts like scales, chords, and rhythm. The instrument’s widespread availability and the abundance of learning resources, such as tutorials and sheet music, make it accessible to aspiring musicians of all ages.",

    # 10 Python text
    "Python’s role in artificial intelligence and machine learning is significant. Frameworks like TensorFlow, Keras, and PyTorch enable developers to create complex AI models efficiently. Python’s simple syntax and extensive documentation make it easier for researchers and developers to prototype and experiment with algorithms. Additionally, Python's ecosystem includes libraries for natural language processing, such as NLTK and spaCy, which help analyze text data. This versatility has cemented Python’s position as the go-to language for AI development.",

    # 11 Sushi text
    "Sushi restaurants, or sushi-ya, are popular dining destinations across the globe. Traditional sushi-ya in Japan often serve sushi directly at the counter, where chefs prepare it in front of customers. This creates a personal and immersive dining experience. Modern sushi restaurants have embraced conveyor belt systems, or kaitenzushi, allowing diners to pick their favorite dishes as they pass by. Whether enjoyed in a fine-dining establishment or a casual eatery, sushi continues to delight food lovers everywhere.",

    # 12 Python text
    "Python is widely regarded as an ideal language for beginners. Its simple and readable syntax allows new programmers to focus on problem-solving rather than grappling with complex syntax. Python’s large and supportive community ensures that beginners have access to tutorials, forums, and documentation. Many educational institutions use Python to introduce students to programming concepts, further solidifying its reputation as a gateway language for future coders.",

    # 13 Sushi text
    "The artistry of sushi lies in its attention to detail. Every element, from the freshness of the fish to the precise slicing techniques, contributes to the dish’s quality. Sushi chefs dedicate years to mastering skills such as forming rice balls and balancing flavors. Each piece of sushi reflects the chef’s expertise and dedication to their craft. This level of artistry has elevated sushi to an iconic status in global cuisine.",

    # 14 Python text
    "Python’s versatility extends to automation and scripting tasks. By writing simple scripts, users can automate repetitive tasks like file management, web scraping, and data entry. Libraries such as BeautifulSoup and Selenium are particularly useful for automating web-based processes. Python’s flexibility allows it to integrate with other tools and systems, making it a powerful choice for workflows across industries ranging from finance to journalism."
]


document_vector = []
for i in range (size_directory):
    f = open(f"Texts/document_{i+1}.txt", "r")
    document_vector.append(f.read())
    f.close()


def NMF_algorithm(n_topics, X):
    """ Algorithm of Lee and Seung's multiplicative update rules for Non-negative Matrix Factorization (NMF).

    Args:
        n_topics (int): Number of topics to identify in the document collection.
        X (int[]): Collection of documents represented as vectors.

    Returns:
        _type_: Matrix W representing the document-topic distribution and matrix H representing the topic-term distribution.
    """
    m = X.shape[0]
    n = X.shape[1]
    
    W = np.random.rand(m, n_topics)
    H = np.random.rand(n_topics, n)
    
    max_iter = 10000
    tol = 1e-10
    norms = []
    epsilon = 1.0e-12
    
    for iter in range(max_iter):
        
        Num_H = W.T @ X
        Denom_H = W.T @ W @ H + epsilon
        H *= Num_H / Denom_H

        Num_W = X @ H.T
        Denom_W = W @ H @ H.T + epsilon
        W *= Num_W / Denom_W

        norm = np.linalg.norm((X - W @ H)/la.norm(X.toarray()), 'fro')
        norms.append(norm)
        if iter > 2 and np.abs(norms[-2] - norms[-1]) < tol:
            print(f"Convergence reached after {iter} iterations.")
            break

    ratio = 100*la.norm(X - W@H)/la.norm(X.toarray())
    
    return W, H, ratio, norms
    
    
    

# Step 1: Preprocess text and build TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document_vector)  # Creates the document-term matrix

# plot matrix X with values noted in the cells
sns.heatmap(X.toarray()[:,:12], annot=True, cmap='coolwarm')
plt.xlabel("Words")
plt.ylabel("Documents")
plt.autoscale()
plt.savefig("heatmap3.png")
plt.show()

n_topics = 2

# Step 2: Apply NMF for topic modeling
def apply_nmf(n_topics, X_input):
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X_input)
    H = nmf.components_
    return W, H, 100*la.norm(X_input - W@H)/la.norm(X_input.toarray())

W, H, ratio, list_norm = NMF_algorithm(n_topics, X)
print(f"X shape = {X.shape} and number of elements = {X.shape[0]*X.shape[1]}")
print(f"total number of elements = {W.shape[0]*W.shape[1] + H.shape[0]*H.shape[1]}")

sns.heatmap(W, annot=True)
plt.xlabel("Topics")
plt.ylabel("Documents")
plt.autoscale()
plt.savefig("Doc_topic3.png")
plt.show()

# Step 3: Interpret Topics
terms = vectorizer.get_feature_names_out()
for i, topic in enumerate(H):
    print("topic", topic[-4:])
    list_words = [terms[j] for j in topic.argsort()[-4:]]
    list_words.reverse()
    # print(f"The topic {i+1} is defined by the words: " + ", ".join(list_words))
print()
documents_by_topic = [[] for _ in range(n_topics)]
for i, topic in enumerate(W):
    documents_by_topic[np.argmax(topic)].append(str(i+1))

for i, topic in enumerate(H):
    break
    # print(f"The documents about {terms[topic.argsort()[-1]]} are the documents: " + ", ".join(documents_by_topic[i]))

ratios = [apply_nmf(i +1, X)[2] for i in range(14)]
keywords_per_topic = {f"Topic {i+1}": terms[np.argsort(H[i, :])[-3:][::-1]].tolist() for i in range(n_topics)}
print(keywords_per_topic)

# Assignation des topics aux documents
topic_documents = {f"Topic {i+1}": [f"Document {j+1}" for j in range(W.shape[0]) if np.argmax(W[j, :]) == i] for i in range(n_topics)}
print(topic_documents)

# Génération des données pour HTML/JS
data = {
    "keywords_per_topic": keywords_per_topic,
    "topic_documents": topic_documents,
    "document_vector": document_vector
}

# Sauvegarde dans un fichier JSON
with open("nmf_results3.json", "w") as f:
    json.dump(data, f, indent=4)

# plt.plot(range(1, 15), ratios, marker='o')
# plt.plot(range(1, 15), [100 - 100*x/14 for x in range(1, 15)], ':')
# plt.xlabel("Number of topics selected (k)")
# plt.ylabel("Ratio of the norms of (X - WH) and (X)")
# plt.legend(["Percentage of recomposition error", "Ideal linear decrease"])
# plt.title("Precision of NMF")
# plt.show()

# plt.plot(range(len(list_norm)), list_norm, ':')
# plt.xlabel("Number of iterations")
# plt.ylabel("Ratio of the norms of (X - WH) and (X)")
# plt.title("Convergence of NMF")
# plt.yrange = [0, 1]
# plt.show()