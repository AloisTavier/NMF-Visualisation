// Charger les données depuis le fichier JSON
document.getElementById('toggle-texts2').addEventListener('click', () => {
    const ImageGraph = document.getElementById('image');
    if (ImageGraph.style.display === 'none') {
        ImageGraph.style.display = 'block';
    } else {
        ImageGraph.style.display = 'none';
    }
});
document.getElementById('toggle-texts3').addEventListener('click', () => {
    const ImageGraph2 = document.getElementById('image2');
    if (ImageGraph2.style.display === 'none') {
        ImageGraph2.style.display = 'block';
    } else {
        ImageGraph2.style.display = 'none';
    }
});
fetch('nmf_results3.json')
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
                    textDiv.innerHTML = `<strong>Text ${index+1}:</strong> ${text}`;
                    textListDiv.appendChild(textDiv);

                });
            } else {
                textListDiv.style.display = 'none';
            }
        });
    })
    .catch(error => console.error('Erreur lors du chargement des données:', error));

