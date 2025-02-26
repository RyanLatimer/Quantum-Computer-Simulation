let gateCount = 1;

function addGate() {
    const gatesDiv = document.getElementById('gates');
    const newGateDiv = document.createElement('div');
    newGateDiv.innerHTML = `
        <label for="gate-type-${gateCount}">Gate Type:</label>
        <select id="gate-type-${gateCount}" name="gates[${gateCount}][type]" onchange="toggleControlQubit(${gateCount})">
            <option value="h">Hadamard</option>
            <option value="cx">CNOT</option>
            <option value="x">Pauli-X</option>
            <option value="y">Pauli-Y</option>
            <option value="z">Pauli-Z</option>
            <option value="phase">Phase</option>
            <option value="t">T</option>
        </select>
        <label for="gate-target-${gateCount}">Target Qubit:</label>
        <input type="number" id="gate-target-${gateCount}" name="gates[${gateCount}][target]" min="0" required>
        <div id="control-qubit-${gateCount}" class="control-qubit" style="display: none;">
            <label for="gate-control-${gateCount}" class="control-label">Control Qubit:</label>
            <input type="number" id="gate-control-${gateCount}" name="gates[${gateCount}][control]" min="0" class="control-input">
        </div>
        <div id="phase-angle-${gateCount}" class="phase-angle" style="display: none;">
            <label for="gate-phase-${gateCount}" class="phase-label">Phase Angle:</label>
            <input type="number" id="gate-phase-${gateCount}" name="gates[${gateCount}][phase]" step="0.01" class="phase-input">
        </div>
    `;
    gatesDiv.appendChild(newGateDiv);
    gateCount++;
}

function toggleControlQubit(index) {
    const gateType = document.getElementById(`gate-type-${index}`).value;
    const controlQubitDiv = document.getElementById(`control-qubit-${index}`);
    const phaseAngleDiv = document.getElementById(`phase-angle-${index}`);
    if (gateType === 'cx') {
        controlQubitDiv.style.display = 'block';
        phaseAngleDiv.style.display = 'none';
    } else if (gateType === 'phase') {
        controlQubitDiv.style.display = 'none';
        phaseAngleDiv.style.display = 'block';
    } else {
        controlQubitDiv.style.display = 'none';
        phaseAngleDiv.style.display = 'none';
    }
}

function applyHadamard(state, qubit, numQubits) {
    const h = [[1, 1], [1, -1]].map(row => row.map(val => val / Math.sqrt(2)));
    const newState = new Array(state.length).fill(math.complex(0, 0));
    for (let i = 0; i < state.length; i++) {
        const bit = (i >> qubit) & 1;
        for (let j = 0; j < 2; j++) {
            const newIndex = i ^ (bit << qubit) ^ (j << qubit);
            newState[newIndex] = math.add(newState[newIndex], math.multiply(h[bit][j], state[i]));
        }
    }
    return newState;
}

function applyCNOT(state, control, target, numQubits) {
    const newState = new Array(state.length).fill(math.complex(0, 0));
    for (let i = 0; i < state.length; i++) {
        if ((i >> control) & 1) {
            const newIndex = i ^ (1 << target);
            newState[newIndex] = state[i];
        } else {
            newState[i] = state[i];
        }
    }
    return newState;
}

function applyPauliX(state, qubit, numQubits) {
    const newState = new Array(state.length).fill(math.complex(0, 0));
    for (let i = 0; i < state.length; i++) {
        const newIndex = i ^ (1 << qubit);
        newState[newIndex] = state[i];
    }
    return newState;
}

function applyPauliY(state, qubit, numQubits) {
    const newState = new Array(state.length).fill(math.complex(0, 0));
    for (let i = 0; i < state.length; i++) {
        const newIndex = i ^ (1 << qubit);
        newState[newIndex] = math.multiply(state[i], (i & (1 << qubit) ? math.complex(0, -1) : math.complex(0, 1)));
    }
    return newState;
}

function applyPauliZ(state, qubit, numQubits) {
    const newState = state.slice();
    for (let i = 0; i < state.length; i++) {
        if (i & (1 << qubit)) {
            newState[i] = math.multiply(newState[i], -1);
        }
    }
    return newState;
}

function applyPhase(state, qubit, numQubits, phase) {
    const newState = state.slice();
    for (let i = 0; i < state.length; i++) {
        if (i & (1 << qubit)) {
            newState[i] = math.multiply(newState[i], math.exp(math.complex(0, phase)));
        }
    }
    console.log(`Phase gate applied with angle ${phase} on qubit ${qubit}:`, newState);
    return newState;
}

function applyT(state, qubit, numQubits) {
    return applyPhase(state, qubit, numQubits, Math.PI / 4);
}

function simulateCircuit(numQubits, gates) {
    let state = new Array(2 ** numQubits).fill(math.complex(0, 0));
    state[0] = math.complex(1, 0); // Start with |0...0> state

    // Apply a sequence of gates to create a more diverse state
    state = applyHadamard(state, 0, numQubits);
    state = applyCNOT(state, 0, 1, numQubits);
    state = applyPauliX(state, 1, numQubits);
    state = applyPhase(state, 0, numQubits, Math.PI / 3);
    state = applyT(state, 1, numQubits);
    state = applyPauliY(state, 0, numQubits);
    state = applyPauliZ(state, 1, numQubits);
    state = applyHadamard(state, 1, numQubits);
    state = applyCNOT(state, 1, 0, numQubits);

    gates.forEach(gate => {
        if (gate.type === 'h') {
            state = applyHadamard(state, gate.target, numQubits);
        } else if (gate.type === 'cx') {
            state = applyCNOT(state, gate.control, gate.target, numQubits);
        } else if (gate.type === 'x') {
            state = applyPauliX(state, gate.target, numQubits);
        } else if (gate.type === 'y') {
            state = applyPauliY(state, gate.target, numQubits);
        } else if (gate.type === 'z') {
            state = applyPauliZ(state, gate.target, numQubits);
        } else if (gate.type === 'phase') {
            state = applyPhase(state, gate.target, numQubits, gate.phase);
        } else if (gate.type === 't') {
            state = applyT(state, gate.target, numQubits);
        }
    });

    const probabilities = state.map(amplitude => math.abs(amplitude) ** 2);
    const result = {
        stateVector: state,
        probabilities: {}
    };
    probabilities.forEach((prob, index) => {
        result.probabilities[index.toString(2).padStart(numQubits, '0')] = prob;
    });

    return result;
}

function plotResults(probabilities) {
    const ctx = document.getElementById('resultsChart').getContext('2d');
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability',
                data: data,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

document.getElementById('circuit-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const numQubits = parseInt(formData.get('num_qubits'));
    const gates = [];
    let isValid = true;
    let errorMessage = '';

    for (let i = 0; i < gateCount; i++) {
        const type = formData.get(`gates[${i}][type]`);
        const target = parseInt(formData.get(`gates[${i}][target]`));
        const control = formData.get(`gates[${i}][control]`);
        const phase = formData.get(`gates[${i}][phase]`);

        if (target >= numQubits || target < 0) {
            isValid = false;
            errorMessage = `Invalid target qubit value for gate ${i + 1}. It must be between 0 and ${numQubits - 1}.`;
            break;
        }

        if (type === 'cx' && (control >= numQubits || control < 0)) {
            isValid = false;
            errorMessage = `Invalid control qubit value for CNOT gate ${i + 1}. It must be between 0 and ${numQubits - 1}.`;
            break;
        }

        const gate = { type, target };
        if (type === 'cx') {
            gate.control = parseInt(control);
        } else if (type === 'phase') {
            gate.phase = parseFloat(phase);
        }
        gates.push(gate);
    }

    if (!isValid) {
        document.getElementById('error-message').textContent = errorMessage;
        document.getElementById('error-message').style.display = 'block';
        return;
    } else {
        document.getElementById('error-message').style.display = 'none';
    }

    const result = simulateCircuit(numQubits, gates);
    document.getElementById('results').textContent = `State Vector: ${JSON.stringify(result.stateVector, null, 2)}\n\nProbabilities: ${JSON.stringify(result.probabilities, null, 2)}`;
    plotResults(result.probabilities);
});