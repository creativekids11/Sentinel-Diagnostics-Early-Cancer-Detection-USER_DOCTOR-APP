// WebSocket connection for real-time reports sync
const socket = new WebSocket(window.location.protocol.replace('http', 'ws') + '//' + window.location.host + '/ws/reports');

socket.onopen = () => {
    console.log('Reports WebSocket connection established');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'report_update') {
        updateReportsTable(data.reports);
    }
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

socket.onclose = () => {
    console.log('Reports WebSocket connection closed');
    // Try to reconnect after 5 seconds
    setTimeout(() => {
        window.location.reload();
    }, 5000);
};

function updateReportsTable(reports) {
    const tbody = document.querySelector('.patient-table tbody');
    if (!tbody) return;

    tbody.innerHTML = reports.map(report => {
        // Format date
        const date = new Date(report.created_at || report.date_generated);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });

        return `
            <tr style="cursor:pointer;">
                <td>R-${String(report.id).padStart(10, '0')}</td>
                <td>${report.patient_name}</td>
                <td>${report.report_title || report.type || 'Medical Report'}</td>
                <td class="status-medium">${report.diagnosis || report.ai_score || 'N/A'}</td>
                <td>${formattedDate}</td>
                <td>
                    <button class="action-btn view-pdf-btn" style="padding: 5px 10px; margin-right: 5px;" onclick="viewPDF(${report.id})">View PDF</button>
                    <button class="action-btn download-pdf-btn" style="padding: 5px 10px; background-color: var(--drac-green);" onclick="downloadPDF(${report.id})">Download PDF</button>
                </td>
            </tr>
        `;
    }).join('');
}

// View PDF function
function viewPDF(reportId) {
    // Open PDF in a new window/tab for viewing
    window.open(`/api/reports/${reportId}/pdf?view=true`, '_blank');
}

// Download PDF function
function downloadPDF(reportId) {
    // Fetch report data to get the title for filename
    fetch(`/api/reports/${reportId}`)
    .then(response => response.json())
    .then(report => {
        if (!report) {
            alert('Report not found');
            return;
        }

        // Create a temporary link to download the PDF with custom filename
        const link = document.createElement('a');
        link.href = `/api/reports/${reportId}/pdf`;
        link.download = `${report.report_title || 'Medical_Report'}.pdf`.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    })
    .catch(error => {
        console.error('Error fetching report:', error);
        alert('Error loading report data');
    });
}

// Function to refresh reports data
function refreshReports() {
    fetch('/api/reports')
        .then(response => response.json())
        .then(reports => {
            updateReportsTable(reports);
        })
        .catch(error => console.error('Error fetching reports:', error));
}

// Initial load
document.addEventListener('DOMContentLoaded', refreshReports);