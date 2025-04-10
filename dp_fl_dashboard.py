import streamlit as st
import time
import random

def simulate_federated_training(num_clients=2, noise_multiplier=1.0, num_rounds=5):
    """
    Simulates per-client and global accuracy & epsilon tracking.
    """
    results = []
    global_epsilon = 0.0
    naive_epsilon = 0.0

    # Simulate per-client accuracy initialization
    client_accuracies = [[random.uniform(0.4, 0.6)] for _ in range(num_clients)]

    for rnd in range(1, num_rounds + 1):
        global_accuracy = 0.0

        # Per-client accuracy updates
        per_client_acc = []
        for c in range(num_clients):
            acc_increment = random.uniform(0.01, 0.03) + (0.005 * rnd) - (0.01 * noise_multiplier)
            new_acc = min(client_accuracies[c][-1] + acc_increment, 1.0)
            client_accuracies[c].append(new_acc)
            per_client_acc.append(new_acc)
            global_accuracy += new_acc

        global_accuracy /= num_clients

        # Epsilon logic
        eps_increment_adv = 0.2 + 0.1 * rnd - 0.05 * noise_multiplier
        eps_increment_naive = 0.5 + 0.2 * rnd

        global_epsilon += max(eps_increment_adv, 0.1)
        naive_epsilon += eps_increment_naive

        results.append({
            "round": rnd,
            "global_accuracy": global_accuracy,
            "epsilon_advanced": global_epsilon,
            "epsilon_naive": naive_epsilon,
            "client_accuracies": per_client_acc
        })

        time.sleep(0.5)

    return results, client_accuracies

def main():
    st.set_page_config(layout="wide")
    st.title("Differential Private Federated Learning Dashboard")

    with st.sidebar:
        st.header("Training Configuration")
        num_clients = st.slider("Number of Clients", 1, 10, 3)
        noise_multiplier = st.slider("Noise Multiplier (σ)", 0.5, 3.0, 1.0, step=0.1)
        num_rounds = st.slider("Number of Rounds", 1, 20, 5)

        st.markdown("---")
        if st.button("🚀 Start Training"):
            st.session_state.run_training = True
        else:
            st.session_state.run_training = st.session_state.get("run_training", False)

    if st.session_state.run_training:
        st.info(f"Running simulation with {num_clients} clients, σ={noise_multiplier}, rounds={num_rounds}...")
        with st.spinner("Training..."):
            results, per_client_acc_history = simulate_federated_training(
                num_clients=num_clients,
                noise_multiplier=noise_multiplier,
                num_rounds=num_rounds
            )
        st.success("Training Complete!")
        st.session_state.run_training = False

        # Extract metrics
        rounds = [r["round"] for r in results]
        global_acc = [r["global_accuracy"] for r in results]
        eps_adv = [r["epsilon_advanced"] for r in results]
        eps_naive = [r["epsilon_naive"] for r in results]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Global Accuracy")
            st.line_chart({"Global Accuracy": global_acc})

        with col2:
            st.subheader("🔐 Epsilon Growth (Advanced vs. Naive)")
            st.line_chart({
                "Advanced Composition": eps_adv,
                "Naive Composition": eps_naive
            })

        st.markdown("---")
        st.subheader("👥 Per-Client Accuracy")

        for client_id, acc_history in enumerate(per_client_acc_history):
            st.line_chart({
                f"Client {client_id + 1}": acc_history
            })

        st.markdown("---")
        st.write("✅ Final Global Accuracy:", round(global_acc[-1], 4))
        st.write("🔐 Final Epsilon (Advanced):", round(eps_adv[-1], 4))
        st.write("🔐 Final Epsilon (Naive):", round(eps_naive[-1], 4))
    else:
        st.write("Use the sidebar to configure and run a training session.")

if __name__ == "__main__":
    main()
