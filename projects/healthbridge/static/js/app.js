/**
 * HealthBridge AI - Main JavaScript Application
 * Handles UI interactions, form validation, and dynamic content
 */

(function() {
    'use strict';

    // HealthBridge Application Namespace
    const HealthBridge = {
        // Current selected mode
        currentMode: null,
        loadingInterval: null,
        loadingProgress: 0,
        loadingStepIndex: 0,
        loadingConfig: null,

        /**
         * Initialize the application
         */
        init: function() {
            this.bindEvents();
            this.checkURLParams();
            window.addEventListener('pageshow', () => {
                this.hideLoading();
            });
        },

        /**
         * Bind all event listeners
         */
        bindEvents: function() {
            // Mode selection cards
            document.querySelectorAll('.mode-card').forEach(card => {
                card.addEventListener('click', (e) => {
                    const mode = card.dataset.mode;
                    this.selectMode(mode);
                });
            });

            // Form submissions
            document.querySelectorAll('form').forEach(form => {
                form.addEventListener('submit', (e) => {
                    this.handleSubmit(e);
                });
            });

            // File input enhancements
            document.querySelectorAll('input[type="file"]').forEach(input => {
                input.addEventListener('change', (e) => {
                    this.handleFileSelect(e);
                });
            });

            // Checkbox groups with auto-save
            document.querySelectorAll('.checkbox-group input').forEach(checkbox => {
                checkbox.addEventListener('change', (e) => {
                    this.saveFormState();
                });
            });

            // Smooth scroll for anchor links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', (e) => {
                    e.preventDefault();
                    const target = document.querySelector(anchor.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });

            // Toggle sections
            document.querySelectorAll('[data-toggle]').forEach(trigger => {
                trigger.addEventListener('click', (e) => {
                    const target = document.querySelector(trigger.dataset.toggle);
                    if (target) {
                        target.classList.toggle('visible');
                    }
                });
            });
        },

        /**
         * Select an analysis mode
         */
        selectMode: function(mode) {
            // Clear previous selections
            document.querySelectorAll('.mode-card').forEach(c => {
                c.classList.remove('selected');
            });
            document.querySelectorAll('.upload-section').forEach(s => {
                s.classList.remove('visible');
            });

            // Select new mode
            const selectedCard = document.getElementById(`card-${mode}`);
            const selectedSection = document.getElementById(`section-${mode}`);

            if (selectedCard) {
                selectedCard.classList.add('selected');
            }
            if (selectedSection) {
                selectedSection.classList.add('visible');
                selectedSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            this.currentMode = mode;

            // Update URL without reloading
            if (window.history.replaceState) {
                window.history.replaceState({}, '', `#${mode}`);
            }

            // Store in session
            sessionStorage.setItem('hb_selected_mode', mode);

            // Trigger analytics event if available
            this.trackEvent('mode_selected', { mode: mode });
        },

        /**
         * Handle form submission
         */
        handleSubmit: function(e) {
            const form = e.target;
            const submitBtn = form.querySelector('.submit-btn');

            // Validate form
            if (!this.validateForm(form)) {
                e.preventDefault();
                return;
            }

            // Show loading
            this.showLoading(form);

            // Disable submit button
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.dataset.originalText = submitBtn.textContent;
                submitBtn.textContent = 'Analyzing...';
            }

            // Clear saved form state on successful submit
            sessionStorage.removeItem('hb_form_state');
        },

        /**
         * Validate form before submission
         */
        validateForm: function(form) {
            let isValid = true;
            const errors = [];

            // Check required file inputs
            form.querySelectorAll('input[type="file"][required]').forEach(input => {
                if (!input.files || input.files.length === 0) {
                    isValid = false;
                    errors.push(`${input.name} is required`);
                    this.highlightError(input);
                } else {
                    this.clearError(input);
                }
            });

            // Validate fasting hours if present
            const fastingInput = form.querySelector('[name="fasting_hours"]');
            if (fastingInput) {
                const hours = parseFloat(fastingInput.value);
                if (hours < 0 || hours > 72) {
                    isValid = false;
                    errors.push('Fasting hours must be between 0 and 72');
                    this.highlightError(fastingInput);
                }
            }

            // Validate sleep hours if present
            const sleepInput = form.querySelector('[name="sleep_hours"]');
            if (sleepInput) {
                const hours = parseFloat(sleepInput.value);
                if (hours < 0 || hours > 24) {
                    isValid = false;
                    errors.push('Sleep hours must be between 0 and 24');
                    this.highlightError(sleepInput);
                }
            }

            // Check file sizes
            form.querySelectorAll('input[type="file"]').forEach(input => {
                if (input.files && input.files[0]) {
                    const file = input.files[0];
                    const maxSize = 50 * 1024 * 1024; // 50MB
                    if (file.size > maxSize) {
                        isValid = false;
                        errors.push(`${file.name} is too large (max 50MB)`);
                        this.highlightError(input);
                    }
                }
            });

            // Show errors if any
            if (!isValid) {
                this.showValidationErrors(errors);
            }

            return isValid;
        },

        /**
         * Highlight form field with error
         */
        highlightError: function(element) {
            element.classList.add('error');
            element.style.borderColor = '#dc2626';

            // Add error message if not present
            let errorMsg = element.parentElement.querySelector('.error-message');
            if (!errorMsg) {
                errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.style.cssText = 'color: #dc2626; font-size: 0.8em; margin-top: 4px;';
                element.parentElement.appendChild(errorMsg);
            }

            if (element.hasAttribute('required') && !element.value) {
                errorMsg.textContent = 'This field is required';
            }
        },

        /**
         * Clear error state from field
         */
        clearError: function(element) {
            element.classList.remove('error');
            element.style.borderColor = '';

            const errorMsg = element.parentElement.querySelector('.error-message');
            if (errorMsg) {
                errorMsg.remove();
            }
        },

        /**
         * Show validation errors
         */
        showValidationErrors: function(errors) {
            // Create or update error container
            let errorContainer = document.getElementById('form-errors');
            if (!errorContainer) {
                errorContainer = document.createElement('div');
                errorContainer.id = 'form-errors';
                errorContainer.className = 'alert alert-error';

                const firstForm = document.querySelector('form');
                if (firstForm) {
                    firstForm.insertBefore(errorContainer, firstForm.firstChild);
                }
            }

            errorContainer.innerHTML = `
                <span class="alert-icon">⚠️</span>
                <div class="alert-content">
                    <strong>Please fix the following errors:</strong>
                    <ul style="margin-top: 8px; margin-bottom: 0; padding-left: 20px;">
                        ${errors.map(e => `<li>${e}</li>`).join('')}
                    </ul>
                </div>
            `;

            // Scroll to errors
            errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        },

        /**
         * Handle file selection
         */
        handleFileSelect: function(e) {
            const input = e.target;
            const file = input.files[0];

            if (file) {
                // Update hint with file info
                const hint = input.parentElement.querySelector('.hint');
                if (hint) {
                    const size = this.formatFileSize(file.size);
                    hint.innerHTML = `<strong>Selected:</strong> ${file.name} (${size})`;
                    hint.style.color = '#059669';
                }

                // Clear error state
                this.clearError(input);

                // Validate file type
                const acceptedTypes = input.accept.split(',').map(t => t.trim());
                const fileExt = '.' + file.name.split('.').pop().toLowerCase();

                if (acceptedTypes.length > 0 && acceptedTypes[0] !== '') {
                    const isAccepted = acceptedTypes.some(type => {
                        if (type.startsWith('.')) {
                            return fileExt === type;
                        }
                        return file.type.match(type.replace('/*', '/'));
                    });

                    if (!isAccepted) {
                        this.showToast(`Warning: ${file.name} may not be a supported file type`, 'warning');
                    }
                }
            }
        },

        /**
         * Format file size for display
         */
        formatFileSize: function(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        /**
         * Show loading overlay
         */
        showLoading: function(form) {
            const overlay = document.getElementById('loading');
            if (overlay) {
                this.configureLoadingOverlay(form);
                overlay.classList.add('visible');
                overlay.setAttribute('aria-hidden', 'false');
                document.body.style.overflow = 'hidden';
            }
        },

        /**
         * Hide loading overlay
         */
        hideLoading: function() {
            const overlay = document.getElementById('loading');
            if (overlay) {
                overlay.classList.remove('visible');
                overlay.setAttribute('aria-hidden', 'true');
                document.body.style.overflow = '';
            }

            if (this.loadingInterval) {
                clearInterval(this.loadingInterval);
                this.loadingInterval = null;
            }

            // Re-enable submit buttons
            document.querySelectorAll('.submit-btn').forEach(btn => {
                btn.disabled = false;
                if (btn.dataset.originalText) {
                    btn.textContent = btn.dataset.originalText;
                }
            });
        },

        /**
         * Configure the loading overlay with mode-aware messaging
         */
        configureLoadingOverlay: function(form) {
            const mode = form?.querySelector('[name="mode"]')?.value || this.currentMode || 'combined';
            const config = this.getLoadingConfig(mode);
            const files = Array.from(form?.querySelectorAll('input[type="file"]') || [])
                .map(input => input.files && input.files[0] ? input.files[0].name : null)
                .filter(Boolean);

            this.loadingConfig = config;
            this.loadingProgress = 8;
            this.loadingStepIndex = 0;

            this.updateLoadingElement('loading-kicker', config.kicker);
            this.updateLoadingElement('loading-title', config.title);
            this.updateLoadingElement('loading-description', config.description);
            this.updateLoadingElement('loading-pill', config.pill);
            this.updateLoadingElement('loading-focus', config.focus);
            this.updateLoadingElement('loading-files', files.length ? files.join(' • ') : 'Preparing uploads');

            this.renderLoadingStages();
            this.renderLoadingProgress();

            if (this.loadingInterval) {
                clearInterval(this.loadingInterval);
            }

            this.loadingInterval = window.setInterval(() => {
                const increment = Math.random() * 8 + 4;
                this.loadingProgress = Math.min(this.loadingProgress + increment, 94);
                this.loadingStepIndex = Math.min(
                    Math.floor((this.loadingProgress / 100) * config.steps.length),
                    config.steps.length - 1
                );
                this.renderLoadingProgress();
            }, 900);
        },

        /**
         * Get mode-specific loading copy
         */
        getLoadingConfig: function(mode) {
            const configs = {
                dna: {
                    kicker: 'Genomic file received',
                    title: 'Reviewing curated variants and building a genetics brief',
                    description: 'HealthBridge is checking the raw genotype file, matching the curated panel, and organizing the strongest follow-up questions.',
                    pill: 'DNA Analysis',
                    focus: 'Genotype interpretation',
                    steps: [
                        { label: 'Validate', detail: 'Checking the raw DNA export and file integrity.' },
                        { label: 'Match', detail: 'Mapping curated SNPs into the internal evidence model.' },
                        { label: 'Prioritize', detail: 'Ranking which variants are strongest and which need phenotype confirmation.' },
                        { label: 'Assemble', detail: 'Rendering the final structured report.' }
                    ]
                },
                blood: {
                    kicker: 'Lab report received',
                    title: 'Normalizing biomarkers into a structured clinical schema',
                    description: 'HealthBridge is extracting lab values, checking ranges, and attaching caveats before the report is rendered.',
                    pill: 'Blood Work Analysis',
                    focus: 'Biomarker normalization',
                    steps: [
                        { label: 'Validate', detail: 'Checking the PDF and collection context for a stable read.' },
                        { label: 'Extract', detail: 'Pulling biomarker values and lab reference ranges from the document.' },
                        { label: 'Contextualize', detail: 'Scoring reliability, optimal ranges, and follow-up tests.' },
                        { label: 'Assemble', detail: 'Packaging the biomarker report for review.' }
                    ]
                },
                combined: {
                    kicker: 'Cross-source workflow started',
                    title: 'Reconciling genotype, biomarkers, and collection context',
                    description: 'HealthBridge is validating both uploads, normalizing them into a shared schema, and looking for the places where the signals actually converge.',
                    pill: 'Combined Analysis',
                    focus: 'Cross-source validation',
                    steps: [
                        { label: 'Validate', detail: 'Checking the DNA file, lab PDF, and collection context.' },
                        { label: 'Normalize', detail: 'Extracting biomarkers and organizing curated genetic findings.' },
                        { label: 'Cross-check', detail: 'Comparing genotype, biomarkers, and context for genuine convergence.' },
                        { label: 'Assemble', detail: 'Building the strongest trustworthy report from the usable evidence.' }
                    ]
                }
            };

            return configs[mode] || configs.combined;
        },

        /**
         * Update a loading element by id
         */
        updateLoadingElement: function(id, text) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = text;
            }
        },

        /**
         * Render loading stages
         */
        renderLoadingStages: function() {
            const list = document.getElementById('loading-stage-list');
            if (!list || !this.loadingConfig) return;

            list.innerHTML = this.loadingConfig.steps.map((step, index) => {
                let state = '';
                if (index < this.loadingStepIndex) {
                    state = 'complete';
                } else if (index === this.loadingStepIndex) {
                    state = 'active';
                }

                return `
                    <div class="loading-stage ${state}">
                        <div class="loading-stage-index">${index + 1}</div>
                        <div class="loading-stage-copy">
                            <strong>${step.label}</strong>
                            <span>${step.detail}</span>
                        </div>
                    </div>
                `;
            }).join('');
        },

        /**
         * Render loading progress and stage copy
         */
        renderLoadingProgress: function() {
            if (!this.loadingConfig) return;

            const currentStep = this.loadingConfig.steps[this.loadingStepIndex];
            const nextStep = this.loadingConfig.steps[Math.min(this.loadingStepIndex + 1, this.loadingConfig.steps.length - 1)];
            const progressFill = document.getElementById('loading-progress-fill');
            const progressLabel = document.getElementById('loading-progress-label');
            const progressPercent = document.getElementById('loading-progress-percent');

            if (progressFill) {
                progressFill.style.width = `${this.loadingProgress}%`;
            }

            if (progressLabel) {
                progressLabel.textContent = `Step ${this.loadingStepIndex + 1} of ${this.loadingConfig.steps.length}`;
            }

            if (progressPercent) {
                progressPercent.textContent = `${Math.round(this.loadingProgress)}%`;
            }

            this.updateLoadingElement('loading-current-step', currentStep.detail);
            this.updateLoadingElement('loading-next-step', nextStep.detail);
            this.renderLoadingStages();
        },

        /**
         * Save form state to sessionStorage
         */
        saveFormState: function() {
            const form = document.querySelector('.upload-section.visible form');
            if (!form) return;

            const formData = new FormData(form);
            const state = {};

            for (let [key, value] of formData.entries()) {
                if (typeof value === 'string') {
                    state[key] = value;
                }
            }

            // Save checkbox states
            form.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                state[cb.name] = cb.checked;
            });

            sessionStorage.setItem('hb_form_state', JSON.stringify(state));
        },

        /**
         * Restore form state from sessionStorage
         */
        restoreFormState: function() {
            const saved = sessionStorage.getItem('hb_form_state');
            if (!saved) return;

            try {
                const state = JSON.parse(saved);
                const form = document.querySelector('.upload-section.visible form');
                if (!form) return;

                Object.entries(state).forEach(([key, value]) => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) {
                        if (input.type === 'checkbox') {
                            input.checked = value;
                        } else if (input.type !== 'file') {
                            input.value = value;
                        }
                    }
                });
            } catch (e) {
                console.error('Error restoring form state:', e);
            }
        },

        /**
         * Check URL parameters for mode selection
         */
        checkURLParams: function() {
            const hash = window.location.hash.slice(1);
            if (hash && ['dna', 'blood', 'combined'].includes(hash)) {
                this.selectMode(hash);
            } else {
                // Check session storage
                const savedMode = sessionStorage.getItem('hb_selected_mode');
                if (savedMode) {
                    this.selectMode(savedMode);
                }
            }
        },

        /**
         * Show toast notification
         */
        showToast: function(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 0.9em;
                z-index: 10000;
                animation: slideIn 0.3s ease;
            `;

            const colors = {
                success: { bg: '#dcfce7', color: '#166534', border: '#bbf7d0' },
                warning: { bg: '#fef3c7', color: '#92400e', border: '#fde68a' },
                error: { bg: '#fee2e2', color: '#991b1b', border: '#fecaca' },
                info: { bg: '#eff6ff', color: '#1e40af', border: '#bfdbfe' }
            };

            const style = colors[type] || colors.info;
            toast.style.background = style.bg;
            toast.style.color = style.color;
            toast.style.border = `1px solid ${style.border}`;
            toast.textContent = message;

            document.body.appendChild(toast);

            setTimeout(() => {
                toast.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        },

        /**
         * Track analytics events
         */
        trackEvent: function(eventName, data = {}) {
            // Google Analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', eventName, data);
            }

            // Plausible
            if (typeof plausible !== 'undefined') {
                plausible(eventName);
            }

            // Console in development
            if (window.location.hostname === 'localhost') {
                console.log('[Analytics]', eventName, data);
            }
        },

        /**
         * Initialize dashboard charts
         */
        initDashboard: function(config) {
            // This would be expanded with Chart.js initialization
            // for trend charts, gauge animations, etc.
            console.log('Dashboard initialized with config:', config);

            // Animate dashboard scores
            document.querySelectorAll('.dashboard-card-score').forEach(el => {
                this.animateNumber(el, 0, parseInt(el.textContent), 1000);
            });
        },

        /**
         * Animate number counting
         */
        animateNumber: function(element, start, end, duration) {
            const startTime = performance.now();

            const update = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);

                // Easing function
                const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                const current = Math.round(start + (end - start) * easeOutQuart);

                element.textContent = current;

                if (progress < 1) {
                    requestAnimationFrame(update);
                }
            };

            requestAnimationFrame(update);
        },

        /**
         * Toggle evidence panel visibility
         */
        toggleEvidence: function(panelId) {
            const panel = document.getElementById(panelId);
            if (panel) {
                panel.classList.toggle('collapsed');
                const toggle = panel.querySelector('.evidence-toggle');
                if (toggle) {
                    toggle.textContent = panel.classList.contains('collapsed')
                        ? 'Show Evidence'
                        : 'Hide Evidence';
                }
            }
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => HealthBridge.init());
    } else {
        HealthBridge.init();
    }

    // Expose to global scope for inline handlers
    window.HealthBridge = HealthBridge;

    // =========================================================================
    // HBTabs — Tab switching for the results report
    // =========================================================================
    window.HBTabs = {
        /**
         * Switch to a named tab panel.
         * @param {string} tabName  e.g. 'overview', 'genetics', 'biomarkers', 'wearables', 'action'
         */
        switchTo: function(tabName) {
            // Deactivate all tabs
            document.querySelectorAll('.report-tab').forEach(function(btn) {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
            });

            // Hide all panels
            document.querySelectorAll('.tab-panel').forEach(function(panel) {
                panel.classList.remove('active');
            });

            // Activate the target tab button
            var targetBtn = document.querySelector('.report-tab[data-tab="' + tabName + '"]');
            if (targetBtn) {
                targetBtn.classList.add('active');
                targetBtn.setAttribute('aria-selected', 'true');
            }

            // Show the target panel
            var targetPanel = document.getElementById('tab-' + tabName);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            // Update URL hash for deep-linking without scrolling
            try {
                history.replaceState(null, '', '#tab-' + tabName);
            } catch(e) {}

            // Scroll to just above the tab bar
            var tabsWrapper = document.querySelector('.report-tabs-wrapper');
            if (tabsWrapper) {
                var offset = tabsWrapper.getBoundingClientRect().top + window.scrollY - 8;
                window.scrollTo({ top: offset, behavior: 'smooth' });
            }
        },

        /**
         * Initialize tabs on page load — reads URL hash or defaults to 'overview'.
         */
        init: function() {
            var hash = window.location.hash;
            if (hash && hash.startsWith('#tab-')) {
                var name = hash.slice(5); // strip '#tab-'
                var exists = document.querySelector('.report-tab[data-tab="' + name + '"]');
                if (exists) {
                    // Use switchTo but without smooth scroll on initial load
                    this.switchTo(name);
                    return;
                }
            }
            // Default: make sure 'overview' is active (it's set active in HTML already,
            // but run switchTo to be safe in case HTML state drifts)
            var first = document.querySelector('.report-tab');
            if (first && first.dataset.tab) {
                var firstTab = first.dataset.tab;
                // Only call if something is already marked active; otherwise trust HTML
                if (!document.querySelector('.report-tab.active')) {
                    this.switchTo(firstTab);
                }
            }
        }
    };

    // Auto-init tabs when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { window.HBTabs.init(); });
    } else {
        window.HBTabs.init();
    }

    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

})();
