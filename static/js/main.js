// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element Selection ---
    const elements = {
        graphParamsForm: document.getElementById('graph-params-form'),
        runButton: document.getElementById('run-button'),
        logContainer: document.getElementById('log-container'),
        graphContainer: document.getElementById('graph-container'),
        modelNameInput: document.getElementById('ollama_model'),
        debugModeCheckbox: document.getElementById('debug_mode'),
        coderDebugModeCheckbox: document.getElementById('coder_debug_mode'),
        downloadLogButton: document.getElementById('download-log-button'),
        perplexityChartCanvas: document.getElementById('perplexityChart').getContext('2d'),
        graphArchitectureSection: document.getElementById('graph-architecture-section'),
        runInterface: document.getElementById('run-interface'),
        chatContainer: document.getElementById('chat-container'),
        chatMessages: document.getElementById('chat-messages'),
        chatForm: document.getElementById('chat-form'),
        chatInput: document.getElementById('chat-input'),
        chatSendButton: document.getElementById('chat-send-button'),
        harvestButton: document.getElementById('harvest-button'),
        diagnosticChatContainer: document.getElementById('diagnostic-chat-container'),
        diagnosticChatMessages: document.getElementById('diagnostic-chat-messages'),
        diagnosticChatForm: document.getElementById('diagnostic-chat-form'),
        diagnosticChatInput: document.getElementById('diagnostic-chat-input'),
        reportContainer: document.getElementById('report-container'),
        downloadReportButton: document.getElementById('download-report-button'),
        codeResultContainer: document.getElementById('code-result-container'),
        codeOutput: document.getElementById('code-output'),
        codeReasoning: document.getElementById('code-reasoning'),
        mbtiGrid: document.querySelector('.mbti-grid'),
    };

    // --- State Management ---
    let state = {
        fullLogContent: '',
        perplexityChart: null,
        perplexityData: { labels: [], values: [] },
        currentSessionId: null,
        eventSource: null,
    };

    const toggleFormElements = (disabled) => {
        const formElements = elements.graphParamsForm.elements;
        for (let i = 0; i < formElements.length; i++) {
            if (formElements[i].id !== 'ollama_model') { // Keep model name display enabled
                formElements[i].disabled = disabled;
            }
        }
    };

    const fetchAndApplyConfig = async () => {
        try {
            const response = await fetch('/config');
            if (!response.ok) {
                console.error('Failed to fetch config from server');
                return;
            }
            const appConfig = await response.json();
    
            if (appConfig.default_model) {
                elements.modelNameInput.value = appConfig.default_model;
            }
            
            if (appConfig.defaults) {
                Object.entries(appConfig.defaults).forEach(([key, value]) => {
                    const formElement = document.getElementById(key);
                    if (key === 'mbti_archetypes' && Array.isArray(value)) {
                        elements.mbtiGrid.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
                        value.forEach(mbti => {
                            const cb = document.getElementById(`mbti-${mbti}`);
                            if (cb) cb.checked = true;
                        });
                    } else if (formElement) {
                        formElement.value = value;
                    }
                });
            }
            console.log("Successfully applied server-side defaults to UI form.");
        } catch (error) {
            console.error('Error applying server configuration:', error);
            addLogMessage(`Error fetching config: ${error.message}`, '#FF5555');
        }
    };

    const MBTI_TYPES = {
        'ISTJ': 'Inspector', 'ISFJ': 'Protector', 'INFJ': 'Advocate', 'INTJ': 'Architect',
        'ISTP': 'Virtuoso', 'ISFP': 'Adventurer', 'INFP': 'Mediator', 'INTP': 'Logician',
        'ESTP': 'Entrepreneur', 'ESFP': 'Entertainer', 'ENFP': 'Campaigner', 'ENTP': 'Debater',
        'ESTJ': 'Executive', 'ESFJ': 'Consul', 'ENFJ': 'Protagonist', 'ENTJ': 'Commander'
    };

    const addLogMessage = (text, color = '#C0C0C0') => {
        state.fullLogContent += text + '\n';
        elements.downloadLogButton.disabled = false;
        const p = document.createElement('p');
        p.textContent = text;
        p.style.color = color;
        p.style.margin = '0';
        p.style.lineHeight = '1.4';
        elements.logContainer.appendChild(p);
        elements.logContainer.scrollTop = elements.logContainer.scrollHeight;
    };
    
    const addChatMessage = (message, sender, container) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${sender}-message`;
        msgDiv.textContent = message;
        container.appendChild(msgDiv);
        container.scrollTop = container.scrollHeight;
        return msgDiv;
    };

    const renderPerplexityChart = () => {
        if (state.perplexityChart) state.perplexityChart.destroy();
        state.perplexityChart = new Chart(elements.perplexityChartCanvas, {
            type: 'line',
            data: {
                labels: state.perplexityData.labels.map(l => `Epoch ${l}`),
                datasets: [{
                    label: 'Average Agent Perplexity',
                    data: state.perplexityData.values,
                    borderColor: 'rgba(255, 75, 75, 1)',
                    backgroundColor: 'rgba(255, 75, 75, 0.2)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    };

    const resetUIForNewRun = () => {
        toggleFormElements(true);
        elements.runButton.textContent = 'Processing...';
        elements.logContainer.innerHTML = '';
        state.fullLogContent = '';
        elements.graphContainer.innerHTML = '<p>ASCII graph will be displayed here...</p>';
        state.perplexityData = { labels: [], values: [] };
        renderPerplexityChart();
        elements.chatContainer.classList.add('hidden');
        elements.reportContainer.classList.add('hidden');
        elements.codeResultContainer.classList.add('hidden');
        elements.runInterface.classList.remove('hidden'); 
        elements.diagnosticChatMessages.innerHTML = '';
        elements.diagnosticChatInput.disabled = true;
        elements.diagnosticChatInput.placeholder = 'Waiting for RAG index...';
    };

    const populateMbtiGrid = () => {
		Object.entries(MBTI_TYPES).forEach(([type, name]) => {
            const id = `mbti-${type}`;
			const optionDiv = document.createElement('div');
            optionDiv.className = 'mbti-option';
			optionDiv.innerHTML = `
				<input type="checkbox" id="${id}" name="mbti_archetypes" value="${type}">
				<label for="${id}">${type} (${name})</label>
			`;
			elements.mbtiGrid.appendChild(optionDiv);
		});
	};

    const handleDebugCheck = () => {
        const isDebug = elements.debugModeCheckbox.checked || elements.coderDebugModeCheckbox.checked;
        elements.modelNameInput.parentElement.classList.toggle('hidden', isDebug);
        
        if (elements.debugModeCheckbox.checked) {
            elements.coderDebugModeCheckbox.checked = false;
        } else if (elements.coderDebugModeCheckbox.checked) {
            elements.debugModeCheckbox.checked = false;
        }
    };

    const handleFormSubmit = async () => {
        if (!elements.graphParamsForm.reportValidity()) return;

        const formData = new FormData(elements.graphParamsForm);
        const params = {};
        formData.forEach((value, key) => {
            if (key === 'mbti_archetypes') {
                if (!params[key]) params[key] = [];
                params[key].push(value);
            } else if (['cot_trace_depth', 'num_questions', 'num_epochs', 'vector_word_size'].includes(key)) {
                params[key] = parseInt(value, 10);
            } else if (['prompt_alignment', 'density', 'learning_rate'].includes(key)) {
                params[key] = parseFloat(value);
            } else if (key.endsWith('mode')) {
                params[key] = value === 'true';
            } else {
                params[key] = value;
            }
        });

        if (!params.mbti_archetypes || params.mbti_archetypes.length < 2) {
            alert('Please select at least 2 MBTI archetypes.');
            return; 
        }

        resetUIForNewRun();
        startLogStream();

        try {
            const response = await fetch('/build_and_run_graph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params })
            });
            const result = await response.json();

            if (!response.ok) {
                throw new Error(`Server Error (${response.status}): ${result.message || 'Unknown error'}\n${result.traceback || ''}`);
            }

            if ('code_solution' in result) {
                elements.codeOutput.textContent = result.code_solution;
                Prism.highlightElement(elements.codeOutput);
                elements.codeReasoning.textContent = result.reasoning;
                elements.codeResultContainer.classList.remove('hidden');
            } else if (result.session_id) {
                state.currentSessionId = result.session_id;
                elements.chatContainer.classList.remove('hidden');
            }

        } catch (error) {
            addLogMessage(`FATAL ERROR: ${error.message}`, '#FF5555');
        } finally {
            toggleFormElements(false);
            elements.runButton.textContent = 'Build and Run Graph';
        }
    };
    
    const handleChat = async (e, input, container, endpoint) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message || !state.currentSessionId) return;
        
        const sendButton = e.target.querySelector('button');
        
        addChatMessage(message, 'user', container);
        input.value = '';
        input.disabled = true;
        if (sendButton) sendButton.disabled = true;

        const aiMsgDiv = addChatMessage('...', 'ai', container);
        aiMsgDiv.textContent = ''; // Clear the placeholder dots immediately

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.currentSessionId, message })
            });
            
            if (!response.body) throw new Error("Response has no body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                aiMsgDiv.textContent += decoder.decode(value, { stream: true });
                container.scrollTop = container.scrollHeight;
            }
        } catch (error) {
            aiMsgDiv.textContent = `Error: ${error.message}`;
        } finally {
            input.disabled = false;
            if(sendButton) sendButton.disabled = false;
            input.focus();
        }
    };

    const handleHarvestClick = async () => {
        if (!state.currentSessionId) return;

        elements.harvestButton.disabled = true;
        elements.harvestButton.textContent = 'Harvesting...';
        elements.chatInput.disabled = true;
        elements.chatSendButton.disabled = true;
        addLogMessage('--- User initiated final harvest. Chat disabled. ---', 'yellow');

        try {
            const response = await fetch('/harvest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.currentSessionId })
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Harvest request failed: ${error.message || 'Server error'}`);
            }

            elements.chatContainer.classList.add('hidden'); // Hide chat on success
            elements.reportContainer.classList.remove('hidden');
            elements.downloadReportButton.onclick = () => {
                window.location.href = `/download_report/${state.currentSessionId}`;
            };
        } catch (error) {
            addLogMessage(`Harvest Error: ${error.message}`, '#FF5555');
            // Re-enable chat if harvest fails
            elements.harvestButton.disabled = false;
            elements.harvestButton.textContent = 'HARVEST REPORT';
            elements.chatInput.disabled = false;
            elements.chatSendButton.disabled = false;
        }
    };

    const startLogStream = () => {
        if (state.eventSource) state.eventSource.close();
        state.eventSource = new EventSource('/stream_log');

        state.eventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                switch (payload.type) {
                    case 'graph':
                        elements.graphContainer.textContent = payload.data;
                        break;
                    case 'perplexity':
                        state.perplexityData.labels.push(payload.data.epoch + 1);
                        state.perplexityData.values.push(payload.data.perplexity);
                        renderPerplexityChart();
                        break;
                    case 'session_id':
                        state.currentSessionId = payload.data;
                        elements.diagnosticChatInput.disabled = false;
                        elements.diagnosticChatInput.placeholder = 'Live RAG diagnostic chat is active.';
                        elements.diagnosticChatContainer.querySelector('button').disabled = false;
                        addLogMessage(`--- DIAGNOSTIC CHAT ENABLED (Session: ${payload.data.substring(0,8)}...) ---`, '#00aaff');
                        break;
                    case 'log':
                    default:
                        addLogMessage(payload.data);
                        break;
                }
            } catch (err) {
                console.warn("Could not parse SSE event, treating as raw log:", event.data);
                addLogMessage(event.data);
            }
        };

        state.eventSource.onerror = (err) => {
            addLogMessage("Log stream connection closed. The run may have finished or an error occurred.", '#FFA500');
            console.error("EventSource failed:", err);
            state.eventSource.close();
        };
    };

    const init = () => {
        populateMbtiGrid();
        fetchAndApplyConfig();
        elements.debugModeCheckbox.addEventListener('change', handleDebugCheck);
        elements.coderDebugModeCheckbox.addEventListener('change', handleDebugCheck);
        elements.runButton.addEventListener('click', handleFormSubmit);
        elements.downloadLogButton.addEventListener('click', () => {
            const blob = new Blob([state.fullLogContent], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `noa-log-${new Date().toISOString()}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
        
        elements.chatForm.addEventListener('submit', (e) => handleChat(e, elements.chatInput, elements.chatMessages, '/chat'));
        elements.diagnosticChatForm.addEventListener('submit', (e) => handleChat(e, elements.diagnosticChatInput, elements.diagnosticChatMessages, '/diagnostic_chat'));
        elements.harvestButton.addEventListener('click', handleHarvestClick);
    };

    init();
});