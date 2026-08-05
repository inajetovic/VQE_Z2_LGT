import pennylane as qml
from pennylane import numpy as np
from hamiltonian import ladder_hamiltonian
from ansatz import circuit
import pandas as pd
import argparse
from scipy.optimize import minimize

def run_vqe(n_plaq, n_layers, J, m, mu, V, charge_inds, n_iter,invariant=True):
    n_fermions = 2*n_plaq + 2
    n_links = 3*n_plaq + 1
    n_qubits = 6*n_plaq + 3

    if invariant:
        s_params = n_links+n_fermions+7*n_plaq+2 #hva
    else:
        s_params = 2*n_links+3*n_fermions+n_plaq

    n_params = n_layers*s_params

    if n_plaq==2:
        ordering = [-1,+1,+1,-1,-1,+1]
    else:
        ordering = [-1,+1,+1,-1,-1,+1,+1,-1]

    #charges = ordering
    for i, c in enumerate(charge_inds):
        sign = ordering[c] 
        ordering[c] = sign * -1

    H = ladder_hamiltonian(n_plaq, J, m, mu, 1, V, ordering)    

    dev = qml.device("default.qubit", wires=n_qubits) 

    @qml.qnode(dev)
    def cost(params):
        circuit(n_plaq, n_layers, n_qubits, H, params, charge_inds, invariant)
        return qml.expval(H)
    
    params =  np.array([np.random.random(1)[0] for _ in range(n_params)], requires_grad=True)
    result = minimize(cost, params, method="SLSQP", options={"maxiter": n_iter})

    return result.fun

def main(n_plaquettes, n_layers, iterations, J, m, mu):
    np.random.seed(0)

    df=pd.DataFrame()
    for n_plaq in n_plaquettes:
        for n_l in n_layers:
                for n_iter in iterations:
                    print(f"Running: P: {n_plaq} L: {n_l} Ni: {n_iter}")
                    for n_j in J:
                        for n_m in m:
                            for n_mu in mu:
                                for flag in [False,True]:
                                    if flag:
                                        circuit="GI"
                                    else:
                                        circuit="ZZ"
                                    
                                    energies=run_vqe(n_plaq, n_l, n_j,n_m,n_mu,0,[],n_iter=n_iter,invariant=flag)
                                    new_row={
                                        "P":n_plaq,
                                        "L":n_l,
                                        "S":1,
                                        "J":n_j,
                                        "m":n_m,
                                        "mu":n_mu,
                                        "Ansatz":circuit,
                                        "E":energies.real
                                    }
                                    print(f"     Res: {new_row}")
                                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) 
    # Save variances_dict to CSV
    csv_output_path = f"slsqp_{n_plaquettes}_layers_{n_layers}_iters_{iterations}_J_{J}_m_{m}_mu_{mu}.csv"
    df.to_csv(csv_output_path, index=False) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSA Bench")
    parser.add_argument('--n_plaquettes', type=int, nargs='+', required=True,
                        help='List of plaquette values (e.g. --n_plaquettes 1 2 3)')
    parser.add_argument('--n_layers', type=int, nargs='+', required=True,
                        help='List of layer values (e.g. --n_layers_list 1 2 3)')
    parser.add_argument('--iterations', type=int,nargs='+',  required=True,
                        help='iterations for optimizer')
    parser.add_argument('--J', type=float,nargs='+',  required=True,
                        help='')
    parser.add_argument('--m', type=float,nargs='+',  required=True,
                        help='')
    parser.add_argument('--mu', type=float,nargs='+',  required=True,
                        help='')
    args = parser.parse_args()
    main(args.n_plaquettes, args.n_layers, args.iterations, args.J, args.m, args.mu)

#python /Users/matteoantonioinajetovic/vqeREV/VQE_Z2_LGT/slsqp_script_multiparams.py --n_plaquettes 1 2  --n_layers 1 2 3  --iterations 200 --J 0 .01 1 5 10 --m .01 1 5 10 --mu .01 1 5 10