// Main JavaScript for ExoHabitatAI 

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = {
        radius: parseFloat(document.getElementById('radius').value),
        mass: parseFloat(document.getElementById('mass').value),
        density: parseFloat(document.getElementById('density').value),
        surface_temp: parseFloat(document.getElementById('surface_temp').value),
        orbital_period: parseFloat(document.getElementById('orbital_period').value),
        distance_from_star: parseFloat(document.getElementById('distance_from_star').value),
        star_type: document.getElementById('star_type').value,
        star_luminosity: parseFloat(document.getElementById('star_luminosity').value),
        star_temp: parseFloat(document.getElementById('star_temp').value),
        metallicity: parseFloat(document.getElementById('metallicity').value)
    };
    
    // Show loading state
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    
    try {
        // Make API call
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        console.log('API Response:', result); // Debug log
        
        if (result.status === 'success' && result.prediction) {
            // Display results
            console.log('Prediction successful, displaying results...');
            displayPredictionResults(result.prediction);
        } else {
            const errorMsg = result.error || 'Unknown error';
            alert('Error: ' + errorMsg);
            console.error('API Error:', result);
            
            // Show error in results section
            const resultsSection = document.getElementById('predictionResults');
            if (resultsSection) {
                resultsSection.style.display = 'block';
                resultsSection.innerHTML = `
                    <div class="alert alert-danger">
                        <h5>Prediction Error</h5>
                        <p>${errorMsg}</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    } finally {
        // Reset button
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
    }
});

function displayPredictionResults(prediction) {
    console.log('Prediction data:', prediction); // Debug log
    
    // Show results section
    const resultsSection = document.getElementById('predictionResults');
    if (!resultsSection) {
        console.error('predictionResults element not found');
        return;
    }
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Update prediction class
    const predictionClass = document.getElementById('predictionClass');
    const classLabel = prediction.class || 'Unknown';
    predictionClass.textContent = classLabel;
    
    // Add color class based on prediction
    predictionClass.className = 'display-6 ';
    if (classLabel.toLowerCase().includes('high')) {
        predictionClass.classList.add('text-success');
    } else if (classLabel.toLowerCase().includes('medium')) {
        predictionClass.classList.add('text-warning');
    } else if (classLabel.toLowerCase().includes('low')) {
        predictionClass.classList.add('text-info');
    } else {
        predictionClass.classList.add('text-danger');
    }
    
    // Update habitability score - ALWAYS display (use default if missing)
    const habitabilityScoreEl = document.getElementById('habitabilityScore');
    const scoreProgressBar = document.getElementById('scoreProgressBar');
    
    // Get habitability score from prediction or calculate default
    let habitabilityScoreValue = prediction.habitability_score;
    
    if (habitabilityScoreValue === undefined || habitabilityScoreValue === null || isNaN(habitabilityScoreValue)) {
        // Use probability as fallback or default
        habitabilityScoreValue = prediction.probability || 0.5;
        console.warn('Habitability score missing, using probability as fallback:', habitabilityScoreValue);
    }
    
    // Ensure it's a number and between 0-1
    habitabilityScoreValue = Math.max(0, Math.min(1, parseFloat(habitabilityScoreValue) || 0.5));
    
    // Convert to percentage
    const habitabilityScorePercent = (habitabilityScoreValue * 100).toFixed(2);
    
    console.log('Displaying Habitability Score:', habitabilityScorePercent + '%'); // Debug log
    
    // Update score display
    if (habitabilityScoreEl) {
        habitabilityScoreEl.textContent = habitabilityScorePercent + '%';
        habitabilityScoreEl.style.display = 'block'; // Ensure visible
    } else {
        console.error('habitabilityScore element not found in DOM');
    }
    
    // Update progress bar
    if (scoreProgressBar) {
        scoreProgressBar.style.width = habitabilityScorePercent + '%';
        scoreProgressBar.setAttribute('aria-valuenow', habitabilityScorePercent);
        scoreProgressBar.style.display = 'block'; // Ensure visible
        
        // Set color based on score
        if (habitabilityScoreValue >= 0.75) {
            scoreProgressBar.className = 'progress-bar bg-success';
        } else if (habitabilityScoreValue >= 0.5) {
            scoreProgressBar.className = 'progress-bar bg-warning';
        } else if (habitabilityScoreValue >= 0.25) {
            scoreProgressBar.className = 'progress-bar bg-info';
        } else {
            scoreProgressBar.className = 'progress-bar bg-danger';
        }
    } else {
        console.error('scoreProgressBar element not found in DOM');
    }
    
    // Update confidence score
    const score = (prediction.probability * 100).toFixed(2);
    document.getElementById('predictionScore').textContent = score + '%';
    
    // Update progress bar for confidence
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = score + '%';
        progressBar.setAttribute('aria-valuenow', score);
        const progressText = document.getElementById('progressText');
        if (progressText) {
            progressText.textContent = score + '%';
        }
    }
    
    // Update progress bar text for habitability score
    const scoreProgressText = document.getElementById('scoreProgressText');
    if (scoreProgressText) {
        scoreProgressText.textContent = habitabilityScorePercent + '%';
    }
    
    // Add details
    const detailsDiv = document.getElementById('predictionDetails');
    if (prediction.probabilities) {
        let detailsHTML = '<h5>Class Probabilities:</h5><ul class="list-group">';
        const classes = ['Non-Habitable', 'Low', 'Medium', 'High'];
        prediction.probabilities.forEach((prob, index) => {
            const className = classes[index] || `Class ${index}`;
            detailsHTML += `<li class="list-group-item d-flex justify-content-between align-items-center">
                ${className}
                <span class="badge bg-primary rounded-pill">${(prob * 100).toFixed(2)}%</span>
            </li>`;
        });
        detailsHTML += '</ul>';
        detailsDiv.innerHTML = detailsHTML;
    }
}

