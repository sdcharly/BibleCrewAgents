<script>
    document.getElementById('verse-form').addEventListener('submit', function(e) {
        e.preventDefault();
        const verse = document.getElementById('verse').value;
        const email = document.getElementById('email').value;
        const statusDiv = document.getElementById('status');

        // Send the data to Flask backend
        fetch('/process_verse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        body: JSON.stringify({ verse: verse, email: email })
        })

        .then(response => response.json())
        .then(data => {
            // Update the status
            statusDiv.innerHTML = 'Result will be sent to your email.';
        })
        .catch((error) => {
            console.error('Error:', error);
            statusDiv.innerHTML = 'An error occurred.';
        });
    });
</script>
