// main.js - Client-side functionality for the Job Matching RAG System

document.addEventListener('DOMContentLoaded', function() {
    // File upload validation
    const resumeInput = document.getElementById('resume');
    if (resumeInput) {
        resumeInput.addEventListener('change', function() {
            const filePath = this.value;
            const allowedExtensions = /(\.pdf|\.txt|\.docx)$/i;
            
            if (!allowedExtensions.exec(filePath)) {
                alert('Please upload a file with .pdf, .txt, or .docx extension');
                this.value = '';
                return false;
            }
        });
    }
    
    // Add active class to current navigation item
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        if (currentLocation === linkPath || 
            (linkPath !== '/' && currentLocation.startsWith(linkPath))) {
            link.classList.add('active');
        }
    });
    
    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});