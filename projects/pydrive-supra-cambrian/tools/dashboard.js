document.addEventListener('DOMContentLoaded', () => {
  const ctx = document.getElementById('fitnessChart').getContext('2d');
  
  // Set Chart.js defaults for dark theme
  Chart.defaults.color = '#949eb2';
  Chart.defaults.font.family = "'Inter', sans-serif";

  const fitnessChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Max Fitness',
          data: [],
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0, 255, 136, 0.1)',
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointBackgroundColor: '#00ff88',
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Avg Fitness',
          data: [],
          borderColor: '#00b8ff',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.3,
          pointBackgroundColor: '#00b8ff',
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(26, 29, 36, 0.9)',
          titleColor: '#fff',
          bodyColor: '#949eb2',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
          }
        },
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
          },
          title: {
            display: true,
            text: 'Generation'
          }
        }
      }
    }
  });

  async function fetchLog() {
    try {
      // Add cache-busting query parameter
      const response = await fetch(`evolution_log.json?t=${new Date().getTime()}`);
      if (!response.ok) throw new Error('Log not found');
      
      const data = await response.json();
      if (data && data.length > 0) {
        updateDashboard(data);
      }
    } catch (err) {
      console.log("Waiting for evolution_log.json...", err.message);
    }
  }

  function updateDashboard(data) {
    // Update Chart
    const labels = data.map(d => `Gen ${d.generation}`);
    const maxFitness = data.map(d => d.max_fitness);
    const avgFitness = data.map(d => d.avg_fitness);

    fitnessChart.data.labels = labels;
    fitnessChart.data.datasets[0].data = maxFitness;
    fitnessChart.data.datasets[1].data = avgFitness;
    fitnessChart.update();

    // Update Leaderboard
    const latestGen = data[data.length - 1];
    document.getElementById('current-gen-badge').innerText = `Generation ${latestGen.generation}`;
    
    const tbody = document.querySelector('#leaderboard-table tbody');
    tbody.innerHTML = ''; // clear

    // Sort by fitness descending (should already be sorted by python, but just in case)
    const sortedPop = [...latestGen.population].sort((a, b) => b.fitness - a.fitness);

    sortedPop.forEach((ind, index) => {
      const tr = document.createElement('tr');
      if (index === 0) tr.classList.add('champion');
      
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td><code>${ind.id}</code></td>
        <td>${ind.fitness.toFixed(2)}</td>
        <td>${ind.mass.toFixed(1)}</td>
        <td>${ind.grip.toFixed(2)}</td>
        <td>${ind.downforce.toFixed(2)}</td>
        <td>${ind.drag.toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Poll every 3 seconds
  setInterval(fetchLog, 3000);
  fetchLog(); // initial fetch
});
