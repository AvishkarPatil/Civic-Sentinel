// Enhanced JavaScript for Civic Sentinel

// Global variables
let currentTheme = localStorage.getItem('theme') || 'light';
let detectionHistory = JSON.parse(localStorage.getItem('detectionHistory')) || [];
let isProcessing = false;

// Initialize application
$(document).ready(function() {
    initializeTheme();
    initializeAnimations();
    initializeKeyboardShortcuts();
    initializeTooltips();
    initializeNotifications();
    initializeProgressTracking();
    
    // Add floating action button
    addFloatingActionButton();
    
    // Initialize page-specific features
    if (window.location.pathname.includes('detect')) {
        initializeDetectionPage();
    }
    
    if (window.location.pathname.includes('analytics')) {
        initializeAnalyticsPage();
    }
});

// Theme Management
function initializeTheme() {
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeToggle();
}

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    updateThemeToggle();
    
    // Animate theme transition
    document.body.style.transition = 'all 0.3s ease';
    setTimeout(() => {
        document.body.style.transition = '';
    }, 300);
}

function updateThemeToggle() {
    const themeToggle = $('#themeToggle');
    if (themeToggle.length) {
        const icon = currentTheme === 'light' ? 'fa-moon' : 'fa-sun';
        const text = currentTheme === 'light' ? 'Dark' : 'Light';
        themeToggle.html(`<i class="fas ${icon} me-1"></i>${text}`);
    }
}

// Animation System
function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe all cards and important elements
    document.querySelectorAll('.card, .stat-card, .alert').forEach(el => {
        observer.observe(el);
    });
    
    // Stagger animations for multiple elements
    $('.card').each(function(index) {
        $(this).css('animation-delay', `${index * 0.1}s`);
    });
}

// Keyboard Shortcuts
function initializeKeyboardShortcuts() {
    $(document).keydown(function(e) {
        // Ctrl/Cmd + D for detect page
        if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            window.location.href = '/detect';
        }
        
        // Ctrl/Cmd + A for analytics page
        if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
            e.preventDefault();
            window.location.href = '/analytics';
        }
        
        // Ctrl/Cmd + H for history page
        if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
            e.preventDefault();
            window.location.href = '/history';
        }
        
        // Ctrl/Cmd + T for theme toggle
        if ((e.ctrlKey || e.metaKey) && e.key === 't') {
            e.preventDefault();
            toggleTheme();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            $('.modal').modal('hide');
            $('.alert-dismissible').alert('close');
        }
    });
}

// Enhanced Tooltips
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Custom tooltips for elements with data-tooltip attribute
    $('.tooltip-custom').hover(
        function() {
            $(this).addClass('tooltip-active');
        },
        function() {
            $(this).removeClass('tooltip-active');
        }
    );
}

// Notification System
function initializeNotifications() {
    window.showNotification = function(message, type = 'info', duration = 5000) {
        const notification = $(`
            <div class="alert alert-${type} alert-dismissible fade show notification-toast" role="alert">
                <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `);
        
        $('#notification-container').append(notification);
        
        // Auto-dismiss after duration
        setTimeout(() => {
            notification.alert('close');
        }, duration);
    };
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Progress Tracking
function initializeProgressTracking() {
    window.updateProgress = function(percentage, message = '') {
        const progressBar = $('.progress-bar');
        const progressText = $('.progress-text');
        
        progressBar.css('width', `${percentage}%`);
        progressBar.attr('aria-valuenow', percentage);
        
        if (message) {
            progressText.text(message);
        }
        
        if (percentage >= 100) {
            setTimeout(() => {
                $('.progress-container').fadeOut();
            }, 1000);
        }
    };
}

// Floating Action Button
function addFloatingActionButton() {
    const fab = $(`
        <button class="fab" id="fabButton" title="Quick Actions">
            <i class="fas fa-plus"></i>
        </button>
    `);
    
    $('body').append(fab);
    
    // FAB menu
    const fabMenu = $(`
        <div class="fab-menu" id="fabMenu" style="display: none;">
            <button class="fab-item" onclick="window.location.href='/detect'">
                <i class="fas fa-search"></i>
                <span>Detect</span>
            </button>
            <button class="fab-item" onclick="toggleTheme()">
                <i class="fas fa-palette"></i>
                <span>Theme</span>
            </button>
            <button class="fab-item" onclick="scrollToTop()">
                <i class="fas fa-arrow-up"></i>
                <span>Top</span>
            </button>
        </div>
    `);
    
    $('body').append(fabMenu);
    
    // FAB click handler
    $('#fabButton').click(function() {
        const menu = $('#fabMenu');
        const isVisible = menu.is(':visible');
        
        if (isVisible) {
            menu.fadeOut(200);
            $(this).find('i').removeClass('fa-times').addClass('fa-plus');
        } else {
            menu.fadeIn(200);
            $(this).find('i').removeClass('fa-plus').addClass('fa-times');
        }
    });
    
    // Hide FAB menu when clicking outside
    $(document).click(function(e) {
        if (!$(e.target).closest('.fab, .fab-menu').length) {
            $('#fabMenu').fadeOut(200);
            $('#fabButton i').removeClass('fa-times').addClass('fa-plus');
        }
    });
}

// Scroll to top function
function scrollToTop() {
    $('html, body').animate({
        scrollTop: 0
    }, 800, 'easeInOutCubic');
}

// Detection Page Features
function initializeDetectionPage() {
    const uploadArea = $('.upload-area');
    const fileInput = $('#fileInput');
    const browseButton = $('#browseButton');
    const uploadIcon = $('#uploadIcon');
    const previewContainer = $('#previewContainer');
    const imagePreview = $('#imagePreview');
    const removeImage = $('#removeImage');
    const analyzeButton = $('#analyzeButton');
    const uploadForm = $('#uploadForm');
    
    // Enhanced drag and drop
    uploadArea.on('dragenter dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).addClass('dragover');
    });
    
    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!$(this).is(e.target) && !$(this).has(e.target).length) {
            $(this).removeClass('dragover');
        }
    });
    
    uploadArea.on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('dragover');
        
        const files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
    
    // Browse button
    browseButton.click(function() {
        fileInput.click();
    });
    
    // File input change
    fileInput.change(function() {
        if (this.files.length > 0) {
            handleFileSelection(this.files[0]);
        }
    });
    
    // Remove image
    removeImage.click(function() {
        resetUploadArea();
    });
    
    // Form submission with progress
    uploadForm.submit(function(e) {
        e.preventDefault();
        
        if (isProcessing) return;
        
        const formData = new FormData(this);
        const file = fileInput[0].files[0];
        
        if (!file) {
            showNotification('Please select an image first', 'warning');
            return;
        }
        
        submitDetectionRequest(formData);
    });
    
    // Paste image from clipboard
    $(document).on('paste', function(e) {
        const items = e.originalEvent.clipboardData.items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const file = item.getAsFile();
                handleFileSelection(file);
                break;
            }
        }
    });
}

// Handle file selection
function handleFileSelection(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showNotification('Please select a valid image file (JPG, PNG, BMP)', 'danger');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size exceeds 16MB limit', 'danger');
        return;
    }
    
    // Create file reader
    const reader = new FileReader();
    reader.onload = function(e) {
        displayImagePreview(e.target.result);
        enableAnalyzeButton();
    };
    reader.readAsDataURL(file);
    
    // Update file input
    const dt = new DataTransfer();
    dt.items.add(file);
    $('#fileInput')[0].files = dt.files;
}

// Display image preview
function displayImagePreview(src) {
    $('#imagePreview').attr('src', src);
    $('#uploadIcon').addClass('d-none');
    $('#previewContainer').removeClass('d-none').addClass('fade-in');
}

// Reset upload area
function resetUploadArea() {
    $('#fileInput').val('');
    $('#uploadIcon').removeClass('d-none');
    $('#previewContainer').addClass('d-none');
    $('#analyzeButton').prop('disabled', true);
    $('.upload-area').removeClass('dragover');
}

// Enable analyze button
function enableAnalyzeButton() {
    $('#analyzeButton').prop('disabled', false).addClass('pulse');
    setTimeout(() => {
        $('#analyzeButton').removeClass('pulse');
    }, 2000);
}

// Submit detection request
function submitDetectionRequest(formData) {
    isProcessing = true;
    
    // Update button state
    const analyzeButton = $('#analyzeButton');
    const originalText = analyzeButton.html();
    analyzeButton.html('<span class="loading-spinner me-2"></span>Analyzing...').prop('disabled', true);
    
    // Show progress container
    showProgressContainer();
    
    // Simulate progress updates
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        updateProgress(progress, getProgressMessage(progress));
    }, 200);
    
    // Submit form
    $.ajax({
        url: '/detect',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            clearInterval(progressInterval);
            updateProgress(100, 'Analysis complete!');
            
            setTimeout(() => {
                if (response.redirect) {
                    window.location.href = response.redirect;
                } else {
                    // Handle inline results
                    displayResults(response);
                }
            }, 1000);
        },
        error: function(xhr, status, error) {
            clearInterval(progressInterval);
            hideProgressContainer();
            showNotification('Analysis failed. Please try again.', 'danger');
            
            // Reset button
            analyzeButton.html(originalText).prop('disabled', false);
            isProcessing = false;
        }
    });
}

// Progress messages
function getProgressMessage(progress) {
    if (progress < 20) return 'Uploading image...';
    if (progress < 40) return 'Preprocessing image...';
    if (progress < 60) return 'Extracting features...';
    if (progress < 80) return 'Running AI analysis...';
    if (progress < 95) return 'Generating results...';
    return 'Finalizing...';
}

// Show/hide progress container
function showProgressContainer() {
    const progressHtml = `
        <div class="progress-container mt-4 fade-in">
            <div class="progress">
                <div class="progress-bar" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            <div class="progress-text text-center text-muted">Starting analysis...</div>
        </div>
    `;
    
    if (!$('.progress-container').length) {
        $('#uploadForm').after(progressHtml);
    }
}

function hideProgressContainer() {
    $('.progress-container').fadeOut(300, function() {
        $(this).remove();
    });
}

// Analytics Page Features
function initializeAnalyticsPage() {
    // Initialize charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }
    
    // Real-time updates
    setInterval(updateAnalytics, 30000); // Update every 30 seconds
    
    // Export functionality
    $('#exportData').click(function() {
        exportAnalyticsData();
    });
}

// Initialize charts
function initializeCharts() {
    // Detection trend chart
    const trendCtx = document.getElementById('trendChart');
    if (trendCtx) {
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Detections',
                    data: [12, 19, 3, 5, 2, 3],
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Accuracy chart
    const accuracyCtx = document.getElementById('accuracyChart');
    if (accuracyCtx) {
        new Chart(accuracyCtx, {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Incorrect'],
                datasets: [{
                    data: [92, 8],
                    backgroundColor: ['#198754', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
}

// Update analytics data
function updateAnalytics() {
    // Fetch latest analytics data
    $.get('/api/analytics', function(data) {
        updateStatCards(data);
        updateCharts(data);
    }).fail(function() {
        console.log('Failed to update analytics');
    });
}

// Update stat cards
function updateStatCards(data) {
    if (data.total_detections) {
        $('.stat-number[data-stat="total"]').text(data.total_detections);
    }
    if (data.accuracy) {
        $('.stat-number[data-stat="accuracy"]').text(data.accuracy + '%');
    }
    if (data.anomalies_found) {
        $('.stat-number[data-stat="anomalies"]').text(data.anomalies_found);
    }
}

// Export analytics data
function exportAnalyticsData() {
    const data = {
        timestamp: new Date().toISOString(),
        detections: detectionHistory,
        summary: {
            total: detectionHistory.length,
            anomalies: detectionHistory.filter(d => d.prediction === 'Anomaly').length,
            normal: detectionHistory.filter(d => d.prediction === 'Normal').length
        }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `civic-sentinel-analytics-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('Analytics data exported successfully', 'success');
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Performance monitoring
function trackPerformance(action, startTime) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    console.log(`${action} took ${duration.toFixed(2)} milliseconds`);
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showNotification('An unexpected error occurred', 'danger');
});

// Service Worker registration (for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Add notification container to body
$(document).ready(function() {
    if (!$('#notification-container').length) {
        $('body').append('<div id="notification-container" style="position: fixed; top: 80px; right: 20px; z-index: 9999; max-width: 400px;"></div>');
    }
});

// Accessibility improvements
$(document).ready(function() {
    // Skip to main content link
    $('body').prepend('<a href="#main-content" class="sr-only sr-only-focusable">Skip to main content</a>');
    
    // Add main content id
    $('main').attr('id', 'main-content');
    
    // Improve focus management
    $('.modal').on('shown.bs.modal', function() {
        $(this).find('[autofocus]').focus();
    });
    
    // Announce dynamic content changes to screen readers
    window.announceToScreenReader = function(message) {
        const announcement = $('<div class="sr-only" aria-live="polite"></div>');
        announcement.text(message);
        $('body').append(announcement);
        setTimeout(() => announcement.remove(), 1000);
    };
});