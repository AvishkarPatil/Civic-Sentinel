// Service Worker for Civic Sentinel PWA
const CACHE_NAME = 'civic-sentinel-v1.0.0';
const urlsToCache = [
    '/',
    '/static/css/style.css',
    '/static/css/fab.css',
    '/static/js/main.js',
    '/static/img/logo.png',
    '/static/img/road-banner.jpg',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js',
    'https://code.jquery.com/jquery-3.6.0.min.js',
    'https://cdn.jsdelivr.net/npm/chart.js'
];

// Install event - cache resources
self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(function(cache) {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', function(event) {
    event.respondWith(
        caches.match(event.request)
            .then(function(response) {
                // Return cached version or fetch from network
                if (response) {
                    return response;
                }
                
                return fetch(event.request).then(function(response) {
                    // Check if we received a valid response
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }
                    
                    // Clone the response
                    var responseToCache = response.clone();
                    
                    caches.open(CACHE_NAME)
                        .then(function(cache) {
                            cache.put(event.request, responseToCache);
                        });
                    
                    return response;
                });
            })
            .catch(function() {
                // Return offline page for navigation requests
                if (event.request.destination === 'document') {
                    return caches.match('/offline');
                }
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.map(function(cacheName) {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Background sync for offline detection submissions
self.addEventListener('sync', function(event) {
    if (event.tag === 'background-sync-detection') {
        event.waitUntil(doBackgroundSync());
    }
});

function doBackgroundSync() {
    return new Promise(function(resolve, reject) {
        // Get pending detections from IndexedDB
        const request = indexedDB.open('CivicSentinelDB', 1);
        
        request.onsuccess = function(event) {
            const db = event.target.result;
            const transaction = db.transaction(['pendingDetections'], 'readonly');
            const store = transaction.objectStore('pendingDetections');
            const getAllRequest = store.getAll();
            
            getAllRequest.onsuccess = function() {
                const pendingDetections = getAllRequest.result;
                
                // Process each pending detection
                const promises = pendingDetections.map(detection => {
                    return fetch('/detect', {
                        method: 'POST',
                        body: detection.formData
                    }).then(response => {
                        if (response.ok) {
                            // Remove from pending detections
                            const deleteTransaction = db.transaction(['pendingDetections'], 'readwrite');
                            const deleteStore = deleteTransaction.objectStore('pendingDetections');
                            deleteStore.delete(detection.id);
                        }
                        return response;
                    });
                });
                
                Promise.all(promises).then(() => resolve()).catch(() => reject());
            };
        };
        
        request.onerror = function() {
            reject();
        };
    });
}

// Push notification handling
self.addEventListener('push', function(event) {
    const options = {
        body: event.data ? event.data.text() : 'New detection result available',
        icon: '/static/img/logo.png',
        badge: '/static/img/logo.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Results',
                icon: '/static/img/view-icon.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/img/close-icon.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('Civic Sentinel', options)
    );
});

// Notification click handling
self.addEventListener('notificationclick', function(event) {
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/analytics')
        );
    } else if (event.action === 'close') {
        // Just close the notification
    } else {
        // Default action - open the app
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handling from main thread
self.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', function(event) {
    if (event.tag === 'update-analytics') {
        event.waitUntil(updateAnalyticsCache());
    }
});

function updateAnalyticsCache() {
    return fetch('/api/analytics')
        .then(response => response.json())
        .then(data => {
            return caches.open(CACHE_NAME).then(cache => {
                return cache.put('/api/analytics', new Response(JSON.stringify(data)));
            });
        })
        .catch(error => {
            console.log('Failed to update analytics cache:', error);
        });
}