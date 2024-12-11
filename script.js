// Charger les données depuis le fichier JSON

const Matrix = document.createElement('img');
Matrix.src = 'heatmap.pdf';
Matrix.width = 700;
Matrix.height = 500;
Matrix.style = 'margin-top: 20px; display: block; margin-right: auto; margin-left: auto;';
document.body.appendChild(Matrix);
fetch('nmf_results.json')
    .then(response => response.json())
    .then(data => {
        const { keywords_per_topic, topic_documents, document_vector } = data;

        // Affichage des mots-clés par topic
        const keywordsDiv = document.getElementById('keywords');
        for (const [topic, keywords] of Object.entries(keywords_per_topic)) {
            const topicDiv = document.createElement('div');
            topicDiv.classList.add('keyword');
            topicDiv.innerHTML = `<strong>${topic}:</strong> ${keywords.join(', ')}`;
            keywordsDiv.appendChild(topicDiv);
        }

        // Affichage des documents par topic
        const docsDiv = document.getElementById('docs');
        for (const [topic, doc] of Object.entries(topic_documents)) {
            const docDiv = document.createElement('div');
            docDiv.classList.add('doc');
            docDiv.innerHTML = `<strong>${topic}:</strong><br> ${doc.join(', ')}`;
            docsDiv.appendChild(docDiv);
        }
        document.getElementById('toggle-texts').addEventListener('click', () => {
            const textListDiv = document.getElementById('text-list');
            if (textListDiv.style.display === 'none') {
                textListDiv.style.display = 'block';
                textListDiv.innerHTML = ''; // Clear any existing content
        
                // Populate with the original texts
                document_vector.forEach((text, index) => {
                    // Create a div containing a text with the title in bold
                    const textDiv = document.createElement('div');
                    textDiv.classList.add('text');
                    textDiv.innerHTML = `<strong>${index}:</strong> ${text}`;
                    textListDiv.appendChild(textDiv);

                });
            } else {
                textListDiv.style.display = 'none';
            }
        });
    })
    .catch(error => console.error('Erreur lors du chargement des données:', error));

