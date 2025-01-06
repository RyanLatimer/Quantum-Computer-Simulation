from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def apply_hadamard(state, qubit):
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return np.kron(np.eye(2**qubit), np.kron(h, np.eye(2**(len(state)//2 - qubit - 1)))) @ state

def apply_cnot(state, control, target):
    size = len(state)
    new_state = np.zeros(size, dtype=complex)
    for i in range(size):
        if (i >> control) & 1:
            new_state[i ^ (1 << target)] = state[i]
        else:
            new_state[i] = state[i]
    return new_state

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    num_qubits = data.get('num_qubits', 2)
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1  # Start with |0...0> state

    # Apply gates to the state
    for gate in data.get('gates', []):
        if gate['type'] == 'h':
            state = apply_hadamard(state, gate['target'])
        elif gate['type'] == 'cx':
            state = apply_cnot(state, gate['control'], gate['target'])

    # Calculate probabilities
    probabilities = np.abs(state)**2
    result = {bin(i)[2:].zfill(num_qubits): prob for i, prob in enumerate(probabilities)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)